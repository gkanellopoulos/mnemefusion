#!/usr/bin/env python3
"""
LoCoMo Evaluation for MnemeFusion

Evaluates MnemeFusion on the LoCoMo (Long-term Conversation Memory) benchmark.

Two evaluation modes:
  Standard (default): Free-text generation + LLM-as-judge.
    Categories 1-4 (single-hop, multi-hop, temporal, open-domain), 1,540 questions.
    Binary CORRECT/WRONG judging with generous matching.
  MCQ (--mcq): 10-choice multiple-choice (deterministic, no LLM judge variance).
    Supplementary/internal mode using Percena/locomo-mc10. Non-standard.

Metrics:
- LLM-as-Judge Accuracy: Binary correctness (standard mode)
- MCQ Accuracy: 10-choice multiple-choice (MCQ mode)
- Recall@K: Fraction of gold evidence in top-K results
- Latency: Query time (p50, p95, p99)

Dataset: LoCoMo — 10 conversations, ~2000 questions
- Categories: Single-hop, Multi-hop, Temporal, Open-domain, Adversarial
- Source: https://github.com/snap-research/locomo

Usage:
    export OPENAI_API_KEY=sk-...

    # Standard evaluation (free-text + LLM judge, categories 1-4)
    python run_eval.py --db-path <path-to.mfdb> --skip-ingestion

    # MCQ evaluation (supplementary, deterministic)
    python run_eval.py --db-path <path-to.mfdb> --skip-ingestion --mcq

    # Multi-run for publication (3 runs, report mean ± stddev)
    python run_eval.py --db-path <path-to.mfdb> --skip-ingestion --runs 3

    # Full ingestion + evaluation
    python run_eval.py --use-llm --llm-model <path-to-model.gguf>

See evals/locomo/README.md for full instructions.
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
import concurrent.futures

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "mnemefusion-python"))
sys.path.insert(0, str(project_root))

# Check dependencies — sentence-transformers is optional when using Rust embedding
# IMPORTANT: Import torch BEFORE mnemefusion on Linux — mnemefusion loads libggml-cuda.so
# which can poison CUDA symbols and break torch's libc10_cuda.so loading.
try:
    import torch  # Must import before sentence_transformers on Windows (DLL search order)
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    print("Install with: pip install openai")
    sys.exit(1)


# =============================================================================
# NScale Cloud Extraction
# =============================================================================

# Extraction system prompt — typed decomposition (ENGRAM-inspired).
# Produces entities, entity_facts, typed records, and entity relationships in one call.
# Research basis: ENGRAM (arXiv 2511.12960) +31 pts from typed separation,
# TReMu (ACL 2025) +47 pts from inferred event dates.
NSCALE_EXTRACTION_SYSTEM = """You extract entities, facts, typed records, and relationships from text into valid JSON.

RECORD TYPES — decompose each turn into 1-3 records:
- episodic: Events, activities, experiences. Set event_date (ISO-8601) when inferrable.
- semantic: Stable facts, preferences, attributes. No event_date.
- procedural: Routines, instructions, recurring patterns. No event_date.

Each record.summary must be a SELF-CONTAINED sentence (understandable without the original text).
Each record.summary must name the subject (e.g., "Alice works at Google", not "Works at Google").

RELATIONSHIPS — extract entity-to-entity links:
- relation_type: spouse, sibling, parent, child, friend, colleague, neighbor, mentor, etc.
- Only extract relationships explicitly stated or strongly implied.

ENTITY FACTS — use the most specific fact_type:
instrument, pet, book, sport, food, hobby, travel, relationship_status, relationship,
family, event, career_goal, location, occupation, affiliation, goal, research_topic,
preference, interest, characteristic, action

Rules:
- instrument: "I play guitar" → instrument "guitar" (NOT interest "playing guitar")
- pet: "My dog Buddy" → pet "dog named Buddy" (NOT preference "dogs")
- book: "I read The Hobbit" → book "The Hobbit" (NOT interest "reading")
- relationship_status: "I'm single" → relationship_status "single" (NOT relationship)
- career_goal: "I want to be a counselor" → career_goal "counseling" (NOT goal)
- family: "I have 3 kids" → family "3 children" (NOT characteristic)
- event: "We went camping" → event "camping trip" (NOT action)
- relationship: ONLY for connections to other people

Keep values concise, under 10 words.

Example 1:
Text: "Alice says: I just got promoted to senior engineer at Google! My husband Bob threw me a surprise party last Saturday."
Session date: 2023-03-15
Output: {"entities":[{"name":"Alice","type":"person"},{"name":"Bob","type":"person"},{"name":"Google","type":"organization"}],"entity_facts":[{"entity":"Alice","fact_type":"occupation","value":"senior engineer","confidence":0.95},{"entity":"Alice","fact_type":"affiliation","value":"Google","confidence":0.95},{"entity":"Alice","fact_type":"relationship","value":"married to Bob","confidence":0.95}],"topics":["career","celebration"],"importance":0.8,"records":[{"record_type":"semantic","summary":"Alice works as a senior engineer at Google","entities":["Alice","Google"]},{"record_type":"episodic","summary":"Alice got promoted to senior engineer","event_date":"2023-03-13","entities":["Alice"]},{"record_type":"episodic","summary":"Bob threw Alice a surprise party for her promotion","event_date":"2023-03-11","entities":["Alice","Bob"]}],"relationships":[{"from_entity":"Alice","to_entity":"Bob","relation_type":"spouse","confidence":0.95}]}

Example 2:
Text: "Mike says: I jog every morning at 6am in Central Park. I've been vegan for three years now."
Output: {"entities":[{"name":"Mike","type":"person"},{"name":"Central Park","type":"location"}],"entity_facts":[{"entity":"Mike","fact_type":"hobby","value":"jogging","confidence":0.95},{"entity":"Mike","fact_type":"location","value":"Central Park","confidence":0.85},{"entity":"Mike","fact_type":"preference","value":"vegan diet","confidence":0.95}],"topics":["exercise","diet"],"importance":0.7,"records":[{"record_type":"procedural","summary":"Mike jogs every morning at 6am in Central Park","entities":["Mike","Central Park"]},{"record_type":"semantic","summary":"Mike has been vegan for three years","entities":["Mike"]}],"relationships":[]}

Example 3:
Text: "Sarah says: We went camping at Yosemite with my college friend Dan two weeks ago. It was amazing!"
Session date: 2023-03-15
Output: {"entities":[{"name":"Sarah","type":"person"},{"name":"Dan","type":"person"},{"name":"Yosemite","type":"location"}],"entity_facts":[{"entity":"Sarah","fact_type":"event","value":"camping at Yosemite","confidence":0.95},{"entity":"Sarah","fact_type":"relationship","value":"college friend Dan","confidence":0.90}],"topics":["outdoors","friendship"],"importance":0.7,"records":[{"record_type":"episodic","summary":"Sarah went camping at Yosemite with her college friend Dan","event_date":"2023-03-01","entities":["Sarah","Dan","Yosemite"]}],"relationships":[{"from_entity":"Sarah","to_entity":"Dan","relation_type":"friend","confidence":0.90}]}"""


class NScaleExtractor:
    """Cloud-based entity extraction via NScale inference API (OpenAI-compatible).

    Uses Qwen3-4B (full-precision) for entity extraction at $0.01/$0.03 per 1M tokens.
    Replaces the local Q4_K_M GGUF model for faster iteration (~1-2h vs 33h re-ingestion).
    """

    def __init__(self, token: str, model: str = "Qwen/Qwen3-4B-Instruct-2507"):
        self.client = OpenAI(
            base_url="https://inference.api.nscale.com/v1",
            api_key=token,
        )
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.errors = 0

    def extract(self, content: str, speaker: str = None, session_date: str = None) -> Optional[dict]:
        """Extract entities, facts, typed records, and relationships using NScale API.

        Args:
            content: The text to extract from (truncated to 1500 chars)
            speaker: Optional speaker name for attribution
            session_date: Optional ISO-8601 date for relative time inference

        Returns:
            Extraction dict matching ExtractionResult schema, or None on failure
        """
        # Truncate content (matching local Qwen3 behavior)
        truncated = content[:1500]

        # Build user message with speaker attribution
        attributed = f'{speaker} says: {truncated}' if speaker else truncated
        speaker_note = (
            f'IMPORTANT: This text was spoken by {speaker}. '
            f'"I"/"my"/"me" = {speaker}. Attribute all facts to {speaker}.\n\n'
        ) if speaker else ''
        date_note = f'\nSession date: {session_date}' if session_date else ''

        user_msg = (
            f'{speaker_note}'
            f'Extract from: "{attributed}"{date_note}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NSCALE_EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            self.total_calls += 1
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

            # Parse response
            text = response.choices[0].message.content
            if not text:
                self.errors += 1
                return None

            # Strip thinking tags if present (Qwen3 sometimes adds them)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            # Use JSONDecoder to parse first JSON object only (handles trailing data)
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(text)

            # Validate schema — ensure all required fields present
            if "entities" not in result:
                result["entities"] = []
            if "entity_facts" not in result:
                result["entity_facts"] = []
            if "topics" not in result:
                result["topics"] = []
            if "importance" not in result:
                result["importance"] = 0.5
            if "records" not in result:
                result["records"] = []
            if "relationships" not in result:
                result["relationships"] = []

            return result

        except Exception as e:
            self.errors += 1
            if self.errors <= 5:
                print(f"    [NScale] Extraction error: {e}")
            return None

    def cost_estimate(self) -> str:
        """Return cost estimate based on token usage."""
        input_cost = self.total_input_tokens * 0.01 / 1_000_000
        output_cost = self.total_output_tokens * 0.03 / 1_000_000
        total = input_cost + output_cost
        return (
            f"NScale: {self.total_calls} calls, "
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out tokens, "
            f"${total:.4f} ({self.errors} errors)"
        )


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
    mrr: float = 0.0

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

    Standard F1 implementation as used in SQuAD evaluations.
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
        context_str = "\n".join([f"- {c}" for c in context[:25]])  # 20 memories + up to 10 profile facts

        prompt = f"""You are a helpful assistant answering questions based on conversation history.

Retrieved memories (dates in brackets show when each conversation occurred):
{context_str}

Question: {question}

Answer the question based on the information in the retrieved memories. Look carefully through ALL the memories for relevant details — the answer may be mentioned briefly or indirectly.
For temporal questions (when did X happen), use the dates in brackets to calculate the answer.
For example, if a memory from [8 May 2023] mentions "yesterday", the event happened on 7 May 2023.
For relative dates like "last Friday" or "the weekend before", calculate the actual date from the memory's date.
For duration questions ("how long"), compute the difference between the relevant dates.
If you find ANY relevant information, provide an answer. Only say "I don't have enough information" if the memories truly contain nothing related to the question.
Keep your answer very concise — ideally 5-6 words."""

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

    def extract_facts(self, question: str, context: List[str], max_tokens: int = 300) -> Tuple[str, int]:
        """
        Two-step RAG: Step 1 — Extract relevant facts from context.

        Reads through all retrieved memories and extracts ONLY the facts
        relevant to answering the question. This separates the "find the
        needle" task from the "reason about it" task.

        Research basis: Chain-of-Note (LongMemEval, +10 pts comprehension),
        EmergenceMem Simple Fast (two-step extraction → answer).

        Returns:
            Tuple of (extracted_facts_text, tokens_used)
        """
        context_str = "\n".join([f"- {c}" for c in context[:25]])

        prompt = f"""Read the following conversation memories and extract ALL facts relevant to answering the question.

Memories:
{context_str}

Question: {question}

Extract every relevant fact, including:
- Names, dates, locations, and specific details
- Relationships between people
- Temporal information (when things happened, durations, sequences)
- Any indirect evidence that could help answer the question

For each relevant memory, write one concise fact. If a memory contains a date, include it.
Skip memories that are completely irrelevant to the question.
If no memories are relevant, write "No relevant facts found."

Relevant facts:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )

            facts = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            self.total_tokens += tokens

            return facts, tokens

        except Exception as e:
            print(f"  [ERROR] Fact extraction failed: {e}")
            return "", 0

    def generate_answer_from_facts(self, question: str, facts: str, max_tokens: int = 150) -> Tuple[str, int]:
        """
        Two-step RAG: Step 2 — Answer from extracted facts.

        Takes pre-extracted facts (from extract_facts) and generates a
        concise answer. The facts are already filtered and organized,
        making reasoning easier for the LLM.

        Returns:
            Tuple of (answer, tokens_used)
        """
        prompt = f"""You are a helpful assistant answering questions based on extracted facts from conversation history.

Relevant facts:
{facts}

Question: {question}

Answer the question based ONLY on the facts above.
For temporal questions (when did X happen), use the dates to calculate the answer.
For duration questions ("how long"), compute the difference between relevant dates.
If the facts contain relevant information, provide an answer. Only say "I don't have enough information" if no facts are relevant.
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

    def select_mcq_answer(self, question: str, context: List[str], choices: List[str]) -> Tuple[int, int]:
        """
        MCQ evaluation: select the best answer from 10 choices given context.

        Replaces generate_answer + judge_answer with a single deterministic call.
        No LLM judge noise — the answer is either correct or not.

        Uses Percena/locomo-mc10 dataset. 10-choice MCQ format
        eliminates ±1-2% LLM judge noise and reduces API cost by 50%.

        Returns:
            Tuple of (selected_choice_index, tokens_used)
        """
        context_str = "\n".join([f"- {c}" for c in context[:25]])

        # Format choices as A-J
        labels = "ABCDEFGHIJ"
        choices_str = "\n".join([f"{labels[i]}. {c}" for i, c in enumerate(choices[:10])])

        prompt = f"""You are answering a multiple-choice question based on conversation history.

Retrieved memories (dates in brackets show when each conversation occurred):
{context_str}

Question: {question}

Choices:
{choices_str}

Select the single best answer based on the retrieved memories. Look carefully through ALL memories for relevant details — the answer may be mentioned briefly or indirectly.
For temporal questions (when did X happen), use the dates in brackets to calculate the answer.
For example, if a memory from [8 May 2023] mentions "yesterday", the event happened on 7 May 2023.
If no memory seems relevant, pick the best guess from the choices.

Answer with ONLY the letter (A-J) of your choice."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )

            answer = response.choices[0].message.content.strip().upper()
            tokens = response.usage.total_tokens
            self.total_tokens += tokens

            # Parse letter to index
            selected_idx = -1
            for char in answer:
                if char in labels:
                    selected_idx = labels.index(char)
                    break

            return selected_idx, tokens

        except Exception as e:
            print(f"  [ERROR] MCQ selection failed: {e}")
            return -1, 0

    def judge_answer(self, question: str, ground_truth: str, prediction: str) -> int:
        """
        Use LLM to judge if the prediction is correct.

        Binary CORRECT/WRONG with generous matching, temporal tolerance,
        and JSON output format.

        Returns:
            1 if correct, 0 if incorrect
        """
        prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

You will be given:
(1) A question about information from past conversations
(2) A 'gold' (ground truth) answer
(3) A generated answer from a memory system

Be generous with your grading — as long as the generated answer touches on the same topic as the gold answer, it should be CORRECT.

For time-related questions, the gold answer will be a specific date/time. The generated answer might use different formats (e.g., "May 7th" vs "7 May") — consider it CORRECT if it refers to the same date or time period.

Question: {question}
Gold answer: {ground_truth}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then return your label as JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content.strip()
            self.total_tokens += response.usage.total_tokens

            # Parse JSON label
            try:
                result = json.loads(raw)
                label = result.get("label", "").upper()
                return 1 if label == "CORRECT" else 0
            except json.JSONDecodeError:
                # Fallback: look for CORRECT/WRONG in raw text
                raw_upper = raw.upper()
                if "CORRECT" in raw_upper and "WRONG" not in raw_upper:
                    return 1
                return 0

        except Exception as e:
            print(f"  [ERROR] Judge failed: {e}")
            return 0


# =============================================================================
# MnemeFusion Integration
# =============================================================================

class MnemeFusionEvaluator:
    """Evaluator for MnemeFusion memory system"""

    # Embedding models that support instruction-based asymmetric encoding.
    # These use prompt_name="query" for queries (asymmetric retrieval).
    INSTRUCTION_MODELS = {
        "Qwen/Qwen3-Embedding-0.6B",
    }

    # Models that need manual text prefixes (not prompt_name-based).
    # Format: model_name -> (query_prefix, document_prefix)
    PREFIX_MODELS = {
        "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    }

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5", use_gpu: bool = True):
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model_name = embedding_model
        self.use_query_prompt = False
        self.query_prefix = ""
        self.doc_prefix = ""

        # Rust embedding mode: if embedding_model is a local directory path, let
        # mnemefusion-core (fastembed/ONNX) handle embedding — no Python ML stack needed.
        self._rust_embed = Path(embedding_model).is_dir()

        if self._rust_embed:
            print(f"  [Embedding] Rust mode: local model dir detected — fastembed will embed in Rust")
            self.embedder = None
            # BGE-base-en-v1.5 produces 768-dim vectors
            self.embedding_dim = 768
        else:
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                print("ERROR: sentence-transformers not installed and embedding_model is not a local path.")
                print("Either install sentence-transformers or pass a local model directory path.")
                sys.exit(1)
            if use_gpu:
                self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True)
            else:
                self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True, device="cpu")
                print(f"  [Embedding] Forced CPU (GPU reserved for LLM extraction)")

            # Detect if model supports instruction-based asymmetric encoding
            self.use_query_prompt = embedding_model in self.INSTRUCTION_MODELS
            if self.use_query_prompt:
                print(f"  [Embedding] Asymmetric mode: queries use prompt_name='query'")

            # Detect if model needs manual text prefixes
            prefixes = self.PREFIX_MODELS.get(embedding_model)
            self.query_prefix = prefixes[0] if prefixes else ""
            self.doc_prefix = prefixes[1] if prefixes else ""
            if prefixes:
                print(f"  [Embedding] Prefix mode: query='{self.query_prefix}', doc='{self.doc_prefix}'")

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
        self.session_map = {}
        self.session_expand = False
        self.neighbor_expand = False

    def create_memory_store(self, db_path: str, use_slm: bool = False, slm_model_path: str = None,
                            use_llm: bool = False, llm_model_path: str = None, llm_tier: str = "balanced",
                            extraction_passes: int = 1, adaptive_k_threshold: float = 0.0,
                            kg_model_path: str = None):
        """Create a new MnemeFusion memory store"""
        self.db_path = db_path

        config = {
            "embedding_dim": self.embedding_dim,
            "entity_extraction_enabled": True,
        }

        # Pass local model path to Rust for fastembed auto-embedding
        if self._rust_embed:
            config["embedding_model"] = self.embedding_model_name
            print(f"  [Embedding] Rust embedding model path: {self.embedding_model_name}")

        if adaptive_k_threshold > 0.0:
            config["adaptive_k_threshold"] = adaptive_k_threshold

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
                self.memory.enable_llm_entity_extraction(llm_model_path, llm_tier, extraction_passes)
                print(f"  [LLM] Enabled {llm_tier} tier extraction (extraction_passes={extraction_passes})")
            except Exception as e:
                print(f"  [LLM] Warning: Failed to enable LLM extraction: {e}")
                print(f"  [LLM] Make sure the wheel was built with --features entity-extraction")

        # Enable Triplex KG extraction if requested (Full tier)
        if kg_model_path:
            try:
                print(f"  [KG] Loading Triplex model: {kg_model_path}")
                self.memory.enable_kg_extraction(kg_model_path)
                print(f"  [KG] Enabled Triplex KG extraction (Full tier)")
            except Exception as e:
                print(f"  [KG] Warning: Failed to enable KG extraction: {e}")
                print(f"  [KG] Make sure the wheel was built with --features entity-extraction")

        # Set embedding function for fact embeddings (enables semantic ProfileSearch)
        # Skip in Rust mode — fastembed handles all embedding internally
        if not self._rust_embed:
            try:
                doc_pfx = self.doc_prefix
                self.memory.set_embedding_fn(lambda text: self.embedder.encode(doc_pfx + text, show_progress_bar=False).tolist())
                print(f"  [Embedding] Fact embedding function set (semantic ProfileSearch enabled)")
            except Exception as e:
                print(f"  [Embedding] Warning: set_embedding_fn not available: {e}")

        # Backfill fact embeddings for existing profiles (one-time, ~4s)
        try:
            n = self.memory.precompute_fact_embeddings()
            print(f"  [Embedding] Precomputed {n} fact embeddings")
        except Exception as e:
            print(f"  [Embedding] Backfill not available: {e}")

        # Rebuild embeddings for first-person content using speaker-aware pronoun substitution.
        # "I joined a gym" -> "Alice joined a gym" for embedding (+0.25 cosine similarity gain
        # with entity-centric queries. One-time backfill, safe to re-run (no-op if already done).
        if not self._rust_embed:
            try:
                n = self.memory.rebuild_speaker_embeddings()
                print(f"  [Embedding] Rebuilt {n} speaker embeddings (1p->3p pronoun substitution)")
            except Exception as e:
                print(f"  [Embedding] Speaker embedding rebuild not available: {e}")

        # Consolidate profiles: remove null values, long values, semantic dedup, garbage entities
        try:
            facts_removed, profiles_deleted = self.memory.consolidate_profiles()
            print(f"  [Consolidation] Removed {facts_removed} facts, deleted {profiles_deleted} profiles")
        except Exception as e:
            print(f"  [Consolidation] Not available: {e}")

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

        # Generate embeddings in batches (raw content, no speaker prefix)
        # Note: Speaker differentiation is handled by the entity scoring dimension
        # (Step 2.1/2.2), not the embedding space. Contextual embeddings were tested
        # in S22 and caused a -12.9pt regression due to query/document mismatch.
        # In Rust mode, skip Python embedding — fastembed handles it in Rust.
        if self._rust_embed:
            all_embeddings = [None] * len(contents)
            print("  [Embedding] Rust mode — skipping Python embedding generation")
        else:
            print("  Generating embeddings...")
            # Batch size 16 for larger models (0.6B+) on 4GB VRAM GPUs.
            # BGE-base (110M) handles 64, but Qwen3-Embedding (0.6B) OOMs at 64.
            batch_size = 16
            all_embeddings = []

            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i:i+batch_size]

                # Apply document prefix for models that need it (e.g., nomic).
                encode_contents = [self.doc_prefix + c for c in batch_contents] if self.doc_prefix else batch_contents
                embeddings = self.embedder.encode(encode_contents, show_progress_bar=False)
                all_embeddings.extend(embeddings.tolist())

                if (i + batch_size) % 500 == 0:
                    print(f"    Embedded {min(i + batch_size, len(contents))}/{len(contents)}")

        # Ingest into MnemeFusion
        print("  Adding to memory store...")

        if use_llm:
            # Use individual add() to enable LLM extraction
            # This is much slower but extracts entity profiles
            #
            # Checkpoint/resume: saves progress to a file so overnight runs
            # survive crashes (llama-cpp GPU crash at ~398 docs).
            # GPU context reset: every 200 docs to prevent memory fragmentation.
            checkpoint_path = self.db_path + ".checkpoint" if self.db_path else None
            completed_indices = set()

            # Load checkpoint if resuming
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path) as f:
                        completed_indices = set(json.load(f))
                    print(f"  [Checkpoint] Resuming from checkpoint: {len(completed_indices)} docs already ingested")
                except Exception:
                    completed_indices = set()

            created_count = 0
            error_count = 0
            skipped_count = len(completed_indices)

            for i, (doc_id, content, metadata) in enumerate(documents):
                # Skip already-ingested docs (checkpoint resume)
                if i in completed_indices:
                    continue

                try:
                    # Parse session_date to Unix timestamp for temporal indexing
                    timestamp = None
                    session_date = metadata.get('session_date', '')
                    if session_date:
                        try:
                            from datetime import datetime
                            # Try common date formats from LoCoMo dataset
                            for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y'):
                                try:
                                    dt = datetime.strptime(session_date.strip(), fmt)
                                    timestamp = dt.timestamp()
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            pass
                    self.memory.add(content, all_embeddings[i], metadata, timestamp=timestamp)
                    created_count += 1
                    completed_indices.add(i)
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"    [WARN] Failed to add doc {i}: {e}")

                # Note: GPU context reset is handled automatically by the ingestion
                # pipeline (every 200 LLM extractions) — no manual call needed.

                # Save checkpoint every 50 docs
                if checkpoint_path and created_count > 0 and created_count % 50 == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(sorted(completed_indices), f)

                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed_so_far = time.time() - start_time
                    done = created_count + skipped_count
                    rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
                    remaining = (len(documents) - done) / rate if rate > 0 else 0
                    print(f"    Progress: {done}/{len(documents)} ({rate:.1f} docs/s, ~{remaining:.0f}s remaining)")

            # Final checkpoint save
            if checkpoint_path:
                with open(checkpoint_path, 'w') as f:
                    json.dump(sorted(completed_indices), f)

            elapsed = time.time() - start_time
            print(f"  [OK] Ingested {created_count} new + {skipped_count} resumed = {created_count + skipped_count} total in {elapsed:.1f}s")
            if error_count > 0:
                print(f"  [WARN] {error_count} errors during ingestion")

            # Clean up checkpoint on successful completion
            if checkpoint_path and len(completed_indices) >= len(documents) - error_count:
                try:
                    os.remove(checkpoint_path)
                    print(f"  [Checkpoint] Removed (ingestion complete)")
                except Exception:
                    pass
        else:
            # Use batch add for fast ingestion (no LLM extraction)
            memories_to_add = []
            for i, (doc_id, content, metadata) in enumerate(documents):
                mem_entry = {
                    "content": content,
                    "metadata": metadata
                }
                # Only include embedding when not in Rust mode
                if all_embeddings[i] is not None:
                    mem_entry["embedding"] = all_embeddings[i]
                # Parse session_date to Unix timestamp for temporal indexing
                session_date = metadata.get('session_date', '')
                if session_date:
                    try:
                        from datetime import datetime
                        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%B %d, %Y', '%d %B %Y'):
                            try:
                                dt = datetime.strptime(session_date.strip(), fmt)
                                mem_entry["timestamp"] = dt.timestamp()
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
                memories_to_add.append(mem_entry)

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

    def retrieve_with_ids(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str], float, List[str]]:
        """
        Retrieve relevant memories with their dialog IDs for Recall@K measurement.

        Returns:
            Tuple of (formatted contents, dialog_ids, latency_ms, profile_context)
        """
        # Generate query embedding — in Rust mode, fastembed handles it internally
        if self._rust_embed:
            query_embedding = None
        else:
            encode_kwargs = {"show_progress_bar": False}
            if self.use_query_prompt:
                encode_kwargs["prompt_name"] = "query"
            query_text = self.query_prefix + query if self.query_prefix else query
            query_embedding = self.embedder.encode([query_text], **encode_kwargs)[0].tolist()

        # Use multi-dimensional query (not just vector search)
        start = time.time()
        intent_info, results, profile_context = self.memory.query(query, query_embedding, top_k)
        latency_ms = (time.time() - start) * 1000

        # Session-level context expansion:
        # If multiple retrieved turns come from the same session, expand to include
        # neighboring turns. Replaces lowest-ranked singletons to stay within budget.
        if self.session_expand and self.session_map:
            results = self._expand_sessions(results)

        # Turn neighborhood expansion: for each retrieved turn, add ±N neighbors
        # from the same session. Addresses the near-miss retrieval pattern where
        # the system finds adjacent turns but not the evidence turn itself.
        # Failure analysis: 73.5% of retrieval failures are near-misses;
        # 56.6% of near-miss evidence is within ±2 turns of a retrieved turn.
        if self.neighbor_expand and self.session_map:
            results = self._expand_neighbors(results)

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

        return contents, dialog_ids, latency_ms, profile_context

    def _expand_sessions(self, results):
        """Expand retrieved results with session context.

        If multiple retrieved turns come from the same conversation session,
        pull in neighboring turns from that session to provide coherent context.
        Expanded turns replace the lowest-ranked singleton results (sessions
        with only 1 hit) to stay within the original result budget.

        Research basis: expanding from individual turn matches to full session
        context improves recall on multi-turn evidence.
        """
        MAX_EXPAND_SESSIONS = 2   # Max sessions to expand
        MIN_SESSION_HITS = 2      # Only expand sessions with >= N retrieved turns
        EXPANSION_WINDOW = 6      # Max new turns to add per session

        # Step 1: Group results by session_key
        session_hits = defaultdict(list)  # session_key -> [position_index]
        for pos, (result_dict, scores_dict) in enumerate(results):
            md = result_dict.get("metadata", {})
            conv_id = md.get("conversation_id", "")
            sess_idx = md.get("session_idx", "")
            if conv_id and sess_idx:
                sk = f"{conv_id}:{sess_idx}"
                session_hits[sk].append(pos)

        # Step 2: Find expansion candidates (>= MIN_SESSION_HITS, exist in session_map)
        candidates = []
        for sk, positions in session_hits.items():
            if len(positions) >= MIN_SESSION_HITS and sk in self.session_map:
                avg_pos = sum(positions) / len(positions)
                candidates.append((sk, len(positions), avg_pos))
        # Sort by hit count desc, then avg position asc (best sessions first)
        candidates.sort(key=lambda x: (-x[1], x[2]))
        expand_keys = set(c[0] for c in candidates[:MAX_EXPAND_SESSIONS])

        if not expand_keys:
            return results

        # Step 3: Collect new turns from expanded sessions (not already retrieved)
        retrieved_dids = set()
        for rd, _ in results:
            did = rd.get("metadata", {}).get("dialog_id", "")
            if did:
                retrieved_dids.add(did)

        new_turns = []  # (session_key, turn_idx, (result_dict, empty_scores))
        for sk in expand_keys:
            added = 0
            for turn_idx, dialog_id, content, speaker, session_date in self.session_map[sk]:
                if dialog_id in retrieved_dids:
                    continue
                if added >= EXPANSION_WINDOW:
                    break
                fake_result = {
                    "content": content,
                    "metadata": {
                        "conversation_id": sk.split(":")[0],
                        "session_idx": sk.split(":")[1],
                        "session_date": session_date,
                        "turn_idx": str(turn_idx),
                        "speaker": speaker,
                        "dialog_id": dialog_id,
                    }
                }
                new_turns.append((sk, turn_idx, (fake_result, {})))
                added += 1

        if not new_turns:
            return results

        # Step 4: Evict lowest-ranked results NOT from expanded sessions
        evictable = []
        for pos, (rd, _) in enumerate(results):
            md = rd.get("metadata", {})
            sk = f"{md.get('conversation_id', '')}:{md.get('session_idx', '')}"
            if sk not in expand_keys:
                evictable.append(pos)
        # Evict from the bottom (highest index = lowest ranked)
        evictable.sort(reverse=True)
        n_evict = min(len(new_turns), len(evictable))
        evict_set = set(evictable[:n_evict])
        new_turns = new_turns[:n_evict]  # Only add as many as we can evict

        if not new_turns:
            return results

        # Step 5: Build session blocks (original hits + expanded turns, sorted by turn_idx)
        session_blocks = defaultdict(list)  # sk -> [(turn_idx, result_tuple)]
        for pos, (rd, sd) in enumerate(results):
            if pos in evict_set:
                continue
            md = rd.get("metadata", {})
            sk = f"{md.get('conversation_id', '')}:{md.get('session_idx', '')}"
            if sk in expand_keys:
                tidx = int(md.get("turn_idx", "0"))
                session_blocks[sk].append((tidx, (rd, sd)))
        for sk, tidx, result_tuple in new_turns:
            session_blocks[sk].append((tidx, result_tuple))
        for sk in session_blocks:
            session_blocks[sk].sort(key=lambda x: x[0])

        # Step 6: Assemble final list
        # - At first hit of each expanded session, insert the full chronological block
        # - Skip subsequent hits from that session (already in the block)
        # - Skip evicted results
        # - Non-expanded results keep their original order
        final = []
        inserted_sessions = set()
        for pos, (rd, sd) in enumerate(results):
            if pos in evict_set:
                continue
            md = rd.get("metadata", {})
            sk = f"{md.get('conversation_id', '')}:{md.get('session_idx', '')}"

            if sk in expand_keys:
                if sk not in inserted_sessions:
                    inserted_sessions.add(sk)
                    for _, rt in session_blocks[sk]:
                        final.append(rt)
                # else: already inserted this session's block
            else:
                final.append((rd, sd))

        return final

    def _expand_neighbors(self, results):
        """Turn neighborhood expansion: add ±N neighbors of each retrieved turn.

        For each retrieved result, finds turns at ±1 and ±2 positions in the
        same session. Adds the best neighbors (closest to highest-ranked results)
        as replacements for the lowest-ranked results.

        Key differences from _expand_sessions:
        - Expands individual results, not sessions (no MIN_SESSION_HITS)
        - Tiny window (±2 turns = 4 neighbors max per result)
        - Prioritizes neighbors of higher-ranked results

        Failure analysis basis: 73.5% of retrieval failures are near-misses
        (right session, wrong turn). 56.6% of near-miss evidence is within
        ±2 turns of a retrieved turn. Upper bound: R@20 48.6% → 71.9%.
        """
        NEIGHBOR_WINDOW = 2    # ±2 turns around each result
        MAX_NEIGHBORS = 5      # Max new turns to add (appended, not replacing)
        EXPAND_TOP_N = 10      # Only expand the top N results (higher-ranked = better signal)

        # Build a turn_idx -> (dialog_id, content, speaker, session_date) lookup per session
        session_turn_lookup = {}  # session_key -> {turn_idx: (dialog_id, content, speaker, session_date)}
        for sk, turns in self.session_map.items():
            session_turn_lookup[sk] = {tidx: (did, content, spk, sd) for tidx, did, content, spk, sd in turns}

        # Collect already-retrieved dialog_ids
        retrieved_dids = set()
        for rd, _ in results:
            did = rd.get("metadata", {}).get("dialog_id", "")
            if did:
                retrieved_dids.add(did)

        # Find candidate neighbors for top-N results
        # Priority score: lower position + closer distance = better
        candidates = []  # (priority, session_key, turn_idx, result_dict)
        for pos, (rd, sd) in enumerate(results[:EXPAND_TOP_N]):
            md = rd.get("metadata", {})
            conv_id = md.get("conversation_id", "")
            sess_idx = md.get("session_idx", "")
            if not conv_id or not sess_idx:
                continue

            sk = f"{conv_id}:{sess_idx}"
            turn_lookup = session_turn_lookup.get(sk, {})
            center_tidx = int(md.get("turn_idx", "0"))

            for offset in range(-NEIGHBOR_WINDOW, NEIGHBOR_WINDOW + 1):
                if offset == 0:
                    continue
                neighbor_tidx = center_tidx + offset
                neighbor = turn_lookup.get(neighbor_tidx)
                if not neighbor:
                    continue

                n_did, n_content, n_speaker, n_session_date = neighbor
                if n_did in retrieved_dids:
                    continue  # Already in results

                # Priority: prefer neighbors of higher-ranked results, prefer closer turns
                priority = pos * 10 + abs(offset)

                fake_result = {
                    "content": n_content,
                    "metadata": {
                        "conversation_id": conv_id,
                        "session_idx": sess_idx,
                        "session_date": n_session_date,
                        "turn_idx": str(neighbor_tidx),
                        "speaker": n_speaker,
                        "dialog_id": n_did,
                    }
                }
                candidates.append((priority, sk, neighbor_tidx, fake_result))
                retrieved_dids.add(n_did)  # Prevent duplicates across results

        if not candidates:
            return results

        # Sort by priority (lower = better) and take top MAX_NEIGHBORS
        candidates.sort(key=lambda x: x[0])
        selected = candidates[:MAX_NEIGHBORS]

        if not selected:
            return results

        # ADDITIVE expansion: keep ALL original results, append neighbors.
        # Previous approach (replacing bottom results) destroyed evidence that
        # was ranked low but still present, causing R@20 to drop.
        # Group neighbors by their source session for chronological insertion
        session_neighbors = defaultdict(list)
        for _, sk, tidx, rd in selected:
            session_neighbors[sk].append((tidx, (rd, {})))

        # Build final list: walk original results, insert neighbors after
        # the last result from each session they belong to
        final = []
        for pos, (rd, sd) in enumerate(results):
            final.append((rd, sd))
            md = rd.get("metadata", {})
            sk = f"{md.get('conversation_id', '')}:{md.get('session_idx', '')}"

            if sk in session_neighbors:
                # Check if the next result is from the same session
                next_same_session = False
                if pos + 1 < len(results):
                    md2 = results[pos + 1][0].get("metadata", {})
                    sk2 = f"{md2.get('conversation_id', '')}:{md2.get('session_idx', '')}"
                    next_same_session = (sk2 == sk)

                if not next_same_session:
                    for tidx, result_tuple in sorted(session_neighbors[sk], key=lambda x: x[0]):
                        final.append(result_tuple)
                    del session_neighbors[sk]

        # Add any remaining neighbors at the end
        for sk, neighbors in session_neighbors.items():
            for tidx, result_tuple in sorted(neighbors, key=lambda x: x[0]):
                final.append(result_tuple)

        return final


# =============================================================================
# Dataset Loading
# =============================================================================

def load_locomo_dataset(dataset_path: str, num_conversations: int = None, include_empty_answers: bool = False) -> Tuple[List[Dict], List[Tuple]]:
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

            if question and (answer or include_empty_answers):
                questions.append((question, answer, category, conv_id, evidence))

    print(f"  [OK] Loaded {len(conversations)} conversations, {len(questions)} questions")

    # Category breakdown
    cat_counts = defaultdict(int)
    for _, _, cat, _, _ in questions:
        cat_counts[cat] += 1
    print(f"  Categories: {dict(cat_counts)}")

    return conversations, questions


def load_mcq_choices(mcq_path: str) -> Dict[str, Dict]:
    """
    Load MCQ choices from Percena/locomo-mc10 dataset (JSONL format).

    Builds a lookup from question text → {choices, correct_choice_index, answer, question_type}.
    The MCQ dataset has 1,986 questions (same as LoCoMo) with 10 answer choices each.

    10-choice MCQ eliminates LLM judge noise (±1-2%),
    makes evaluation deterministic, and reduces API cost by ~50%.
    """
    if not os.path.exists(mcq_path):
        print(f"ERROR: MCQ dataset not found at {mcq_path}")
        print("Download with: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('Percena/locomo-mc10')\"")
        print("Then provide the path to the locomo_mc10.json JSONL file.")
        sys.exit(1)

    mcq_map = {}  # keyed by question_id (e.g., "conv-26_q0") for unique matching
    mcq_by_text = {}  # fallback: keyed by question text
    with open(mcq_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            entry = {
                'choices': item['choices'],
                'correct_choice_index': item['correct_choice_index'],
                'answer': item['answer'],
                'question_type': item['question_type'],
                'question_id': item['question_id'],
            }
            mcq_map[item['question_id']] = entry
            mcq_by_text[item['question']] = entry  # last-wins for 12 duplicates

    print(f"  [MCQ] Loaded {len(mcq_map)} by question_id, {len(mcq_by_text)} by text from {mcq_path}")
    # Return text-based map for eval loop (question text is our lookup key).
    # 12/1986 duplicates (0.6%) use last-wins — acceptable ambiguity.
    return mcq_by_text


def prepare_documents(conversations: List[Dict]) -> Tuple[List[Tuple[str, str, Dict]], Dict[str, List]]:
    """
    Prepare conversation turns as documents for ingestion.

    Returns:
        Tuple of:
        - List of (doc_id, content, metadata) tuples
        - session_map: "conv_id:session_idx" -> [(turn_idx, dialog_id, content, speaker, session_date)]
    """
    documents = []
    session_map = defaultdict(list)

    for conv in conversations:
        conv_id = conv.get('sample_id', 'unknown')
        conv_data = conv.get('conversation', {})

        session_idx = 1
        while f'session_{session_idx}' in conv_data:
            session_key = f'session_{session_idx}'
            session_date = conv_data.get(f'session_{session_idx}_date_time', f'session_{session_idx}')
            turns = conv_data.get(session_key, [])
            map_key = f"{conv_id}:{session_idx}"

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
                session_map[map_key].append((turn_idx, dialog_id, text, speaker, session_date))

            session_idx += 1

    # Sort each session's turns by turn_idx (should already be ordered, but ensure)
    for key in session_map:
        session_map[key].sort(key=lambda x: x[0])

    return documents, dict(session_map)


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
    kg_model_path: str = None,
    output_path: str = None,
    verbose: bool = False,
    db_path: str = None,
    skip_ingestion: bool = False,
    extraction_passes: int = 1,
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    nscale_extract: bool = False,
    nscale_token: str = None,
    nscale_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    session_expand: bool = False,
    neighbor_expand: bool = False,
    two_step_rag: bool = False,
    mcq: bool = False,
    mcq_path: str = None,
    adaptive_k_threshold: float = 0.0,
    runs: int = 1,
    retrieval_only: bool = False,
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
        nscale_extract: Use NScale cloud API for extraction
        nscale_token: NScale service token
        nscale_model: NScale model name

    Returns:
        EvaluationResults with all metrics
    """
    print("=" * 70)
    print("MnemeFusion LoCoMo Evaluation")
    print("=" * 70)
    print(f"\nMethodology:")
    if mcq:
        print(f"  Mode:             MCQ (10-choice, deterministic, non-standard)")
    else:
        print(f"  Mode:             Free-text + LLM-as-judge (standard protocol)")
    print(f"  Answer model:     GPT-4o-mini, temperature=0")
    print(f"  Judge model:      GPT-4o-mini, temperature=0")
    if mcq:
        print(f"  Scoring:          MCQ — correct if selected choice matches ground truth")
    else:
        print(f"  Scoring:          Binary CORRECT/WRONG (standard judge prompt)")
    cat_list = categories if categories else [1, 2, 3, 4]
    print(f"  Categories:       {cat_list}")
    print(f"  Embedding model:  {embedding_model}")
    if nscale_extract:
        print(f"  Extraction:       NScale Cloud ({nscale_model})")
    elif use_llm:
        tier_label = "Full (Phi-4 + Triplex)" if kg_model_path else f"{llm_tier}"
        print(f"  Extraction:       Native LLM ({tier_label} tier, {extraction_passes} pass{'es' if extraction_passes > 1 else ''})")
        if kg_model_path:
            print(f"  KG Extraction:    Triplex ({kg_model_path})")
    elif use_slm:
        print(f"  Extraction:       Python SLM (0.6B)")
    else:
        print(f"  Extraction:       Disabled (baseline)")
    if session_expand:
        print(f"  Session expand:   ENABLED")
    if neighbor_expand:
        print(f"  Neighbor expand:  ENABLED")
    if two_step_rag:
        print(f"  Two-step RAG:     ENABLED")
    if adaptive_k_threshold > 0.0:
        print(f"  Adaptive-K:       threshold={adaptive_k_threshold}")
    if runs > 1:
        print(f"  Runs:             {runs} (will report mean ± stddev)")
    print("=" * 70)

    # Load MCQ choices if MCQ mode enabled
    mcq_map = {}
    if mcq:
        if not mcq_path:
            # Auto-detect from HuggingFace cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--Percena--locomo-mc10")
            candidates = []
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if f == "locomo_mc10.json":
                        candidates.append(os.path.join(root, f))
            if candidates:
                # Prefer the 'data/' version (JSONL with MCQ fields)
                mcq_path = sorted(candidates, key=lambda p: 'data' in p, reverse=True)[0]
                print(f"  [MCQ] Auto-detected dataset at {mcq_path}")
            else:
                print("ERROR: MCQ dataset not found in HuggingFace cache.")
                print("Download with: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('Percena/locomo-mc10')\"")
                sys.exit(1)
        mcq_map = load_mcq_choices(mcq_path)

    # Load dataset (MCQ mode includes adversarial questions with empty answers)
    conversations, all_questions = load_locomo_dataset(dataset_path, num_conversations, include_empty_answers=mcq)

    # Filter questions by category if specified
    if categories:
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in categories]
    else:
        # Default: categories 1-4 (standard LoCoMo protocol, 1,540 questions)
        # Category 5 (adversarial) excluded by default — pass --categories 5 to include
        questions = [(q, a, c, cid, e) for q, a, c, cid, e in all_questions if c in [1, 2, 3, 4]]

    if max_questions:
        questions = questions[:max_questions]

    print(f"\nEvaluating {len(questions)} questions")

    # Prepare documents
    documents, session_map = prepare_documents(conversations)
    print(f"Prepared {len(documents)} documents ({len(session_map)} sessions)")

    # Initialize components
    # When LLM extraction is active, keep embedder on CPU so LLM gets full GPU VRAM
    embed_gpu = not use_llm
    evaluator = MnemeFusionEvaluator(embedding_model=embedding_model, use_gpu=embed_gpu)
    evaluator.session_map = session_map
    evaluator.session_expand = session_expand
    evaluator.neighbor_expand = neighbor_expand
    if session_expand:
        print(f"  [Session Expand] Enabled — will expand sessions with >= 2 retrieved turns")
    if neighbor_expand:
        print(f"  [Neighbor Expand] Enabled — will add ±2 neighboring turns for each retrieved result")
    llm = None if retrieval_only else LLMClient(model="gpt-4o-mini")

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
            llm_tier=llm_tier,
            extraction_passes=extraction_passes,
            adaptive_k_threshold=adaptive_k_threshold,
            kg_model_path=kg_model_path if _ingest_now else None
        )

        if _ingest_now:
            if nscale_extract:
                # Two-phase ingestion: batch add (fast) + NScale extraction (cloud)
                print("\n  [NScale] Phase 1: Batch ingestion (embeddings only)...")
                ingestion_time = evaluator.ingest_documents(documents, use_llm=False)

                print(f"\n  [NScale] Phase 2: Cloud extraction via {nscale_model}...")
                nscale = NScaleExtractor(token=nscale_token, model=nscale_model)
                extract_start = time.time()

                # Get all memory IDs from the store (ordered by insertion)
                all_memories = evaluator.memory.get_recent(len(documents))
                # Build a map from content -> memory_id for matching
                content_to_mid = {}
                for mem_dict, _ts in all_memories:
                    content_to_mid[mem_dict["content"]] = mem_dict["id"]

                # Phase 2a: Extract concurrently via NScale API (IO-bound)
                NSCALE_WORKERS = 8  # Concurrent API calls
                print(f"    Using {NSCALE_WORKERS} concurrent workers...")

                def _extract_one(args):
                    idx, content, speaker, session_date = args
                    return idx, nscale.extract(content, speaker=speaker, session_date=session_date)

                extract_tasks = []
                for i, (doc_id, content, metadata) in enumerate(documents):
                    speaker = metadata.get("speaker")
                    session_date = metadata.get("session_date")
                    extract_tasks.append((i, content, speaker, session_date))

                extraction_results = [None] * len(documents)
                with concurrent.futures.ThreadPoolExecutor(max_workers=NSCALE_WORKERS) as pool:
                    futures = {pool.submit(_extract_one, t): t[0] for t in extract_tasks}
                    done_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        idx, result = future.result()
                        extraction_results[idx] = result
                        done_count += 1
                        if done_count % 500 == 0:
                            elapsed = time.time() - extract_start
                            rate = done_count / elapsed
                            remaining = (len(documents) - done_count) / rate if rate > 0 else 0
                            print(f"    Extracted: {done_count}/{len(documents)} ({rate:.1f} docs/s, ~{remaining:.0f}s remaining)")

                extract_api_time = time.time() - extract_start
                print(f"    API calls done in {extract_api_time:.1f}s")

                # Phase 2b: Apply extraction results to profiles (sequential, Rust-side)
                extracted = 0
                failed = 0
                for i, (doc_id, content, metadata) in enumerate(documents):
                    memory_id = content_to_mid.get(content)
                    if not memory_id:
                        failed += 1
                        continue

                    result = extraction_results[i]
                    if result and (result.get("entities") or result.get("entity_facts")
                                   or result.get("records") or result.get("relationships")):
                        try:
                            evaluator.memory.apply_extraction(memory_id, result)
                            extracted += 1
                        except Exception as e:
                            failed += 1
                            if failed <= 5:
                                print(f"    [NScale] apply_extraction error: {e}")
                    else:
                        failed += 1

                extract_elapsed = time.time() - extract_start
                ingestion_time += extract_elapsed
                print(f"  [NScale] Extraction complete: {extracted} extracted, {failed} failed in {extract_elapsed:.1f}s")
                print(f"  [NScale] {nscale.cost_estimate()}")

                # Post-ingestion cleanup (same as local LLM path)
                try:
                    facts_removed, profiles_deleted = evaluator.memory.consolidate_profiles()
                    print(f"  [Post-ingestion] Consolidated: removed {facts_removed} facts, deleted {profiles_deleted} profiles")
                except Exception as e:
                    print(f"  [Post-ingestion] Consolidation error: {e}")

                try:
                    n_summarized = evaluator.memory.summarize_profiles()
                    print(f"  [Post-ingestion] Summarized {n_summarized} profiles")
                except Exception as e:
                    print(f"  [Post-ingestion] Summary error: {e}")

            else:
                # Standard ingestion (local LLM or no extraction)
                ingestion_time = evaluator.ingest_documents(documents, use_llm=use_llm)

                # Post-ingestion: consolidate profiles and generate summaries
                if use_llm:
                    try:
                        facts_removed, profiles_deleted = evaluator.memory.consolidate_profiles()
                        print(f"  [Post-ingestion] Consolidated: removed {facts_removed} facts, deleted {profiles_deleted} profiles")
                    except Exception as e:
                        print(f"  [Post-ingestion] Consolidation error: {e}")

                    try:
                        n_summarized = evaluator.memory.summarize_profiles()
                        print(f"  [Post-ingestion] Summarized {n_summarized} profiles")
                    except Exception as e:
                        print(f"  [Post-ingestion] Summary error: {e}")
        else:
            print(f"  [SKIP] Using existing DB at {db_path} — skipping ingestion")
            ingestion_time = 0.0

        # Multi-run support: wrap evaluation in outer loop
        run_accuracies = []

        for run_idx in range(runs):
            if runs > 1:
                print(f"\n{'#' * 70}")
                print(f"# RUN {run_idx + 1}/{runs}")
                print(f"{'#' * 70}")

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
                retrieved_content, retrieved_ids, retrieval_latency, profile_ctx = evaluator.retrieve_with_ids(question, top_k=recall_k)
                latencies.append(retrieval_latency)

                # Calculate Recall@K (Solution 6)
                # Check if evidence dialog IDs appear in retrieved results at various K
                evidence_set = set(evidence) if evidence else set()
                r_at_5 = 0.0
                r_at_10 = 0.0
                r_at_20 = 0.0
                if evidence_set:
                    # All retrieved_ids are now real memory IDs (no synthetic profile facts)
                    # With neighbor expansion, retrieved_ids may include additive neighbors
                    # beyond position 20. R@K uses the first K from the original ranking,
                    # but R@20 extends to include all IDs (original + neighbors).
                    found_at_5 = len(evidence_set & set(retrieved_ids[:5]))
                    found_at_10 = len(evidence_set & set(retrieved_ids[:10]))
                    found_at_20 = len(evidence_set & set(retrieved_ids[:min(20, len(retrieved_ids))]))
                    r_at_5 = found_at_5 / len(evidence_set)
                    r_at_10 = found_at_10 / len(evidence_set)
                    r_at_20 = found_at_20 / len(evidence_set)

                # Build context: profile context (entity knowledge) + real memories
                # Profile context helps entity-dependent queries (single-hop, temporal)
                # but hurts open-domain by displacing real memories. Intent-conditional
                # inclusion would be ideal but requires more research.
                # With neighbor expansion, retrieved_content may have up to top_k + 5
                # items (original + additive neighbors). Include all to avoid cutting
                # off neighbors that were specifically selected for near-miss evidence.
                n_profile = min(len(profile_ctx), 5)
                content_budget = top_k - n_profile
                if neighbor_expand:
                    content_budget = len(retrieved_content)  # Include all (original + neighbors)
                context_with_facts = profile_ctx[:n_profile] + retrieved_content[:content_budget]

                # Generate answer (or select MCQ choice)
                gen_start = time.time()
                if retrieval_only:
                    answer = ""
                    tokens = 0
                    judge_score = 0
                    f1 = 0.0
                    bleu = 0.0
                elif mcq and question in mcq_map:
                    # MCQ mode: single LLM call to select from 10 choices
                    mcq_data = mcq_map[question]
                    selected_idx, tokens = llm.select_mcq_answer(
                        question, context_with_facts, mcq_data['choices']
                    )
                    judge_score = 1 if selected_idx == mcq_data['correct_choice_index'] else 0
                    answer = mcq_data['choices'][selected_idx] if 0 <= selected_idx < len(mcq_data['choices']) else "INVALID"
                    f1 = calculate_f1_score(answer, ground_truth)
                    bleu = calculate_bleu_score(answer, ground_truth)
                elif two_step_rag:
                    # Two-step RAG: (1) extract relevant facts, (2) answer from facts
                    facts, extract_tokens = llm.extract_facts(question, context_with_facts)
                    answer, answer_tokens = llm.generate_answer_from_facts(question, facts)
                    tokens = extract_tokens + answer_tokens
                    judge_score = llm.judge_answer(question, ground_truth, answer)
                    f1 = calculate_f1_score(answer, ground_truth)
                    bleu = calculate_bleu_score(answer, ground_truth)
                else:
                    answer, tokens = llm.generate_answer(question, context_with_facts)
                    judge_score = llm.judge_answer(question, ground_truth, answer)
                    f1 = calculate_f1_score(answer, ground_truth)
                    bleu = calculate_bleu_score(answer, ground_truth)
                gen_latency = (time.time() - gen_start) * 1000

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
                    if retrieval_only:
                        print(f"    {recall_str} | R@20: {r_at_20:.0%}")
                    elif mcq and question in mcq_map:
                        print(f"    MCQ: {'CORRECT' if judge_score else 'WRONG'} (selected: {answer[:50]}) | {recall_str}")
                    else:
                        print(f"    Judge: {'CORRECT' if judge_score else 'WRONG'} | F1: {f1:.2f} | BLEU: {bleu:.2f} | {recall_str}")

            eval_time = time.time() - eval_start
            print("-" * 70)

            # Aggregate results
            final_results = EvaluationResults(
                total_questions=len(results),
                num_documents=len(documents),
                num_conversations=len(conversations),
                total_ingestion_time_s=ingestion_time,
                total_evaluation_time_s=eval_time,
                total_tokens_used=0 if retrieval_only else llm.total_tokens
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

                    # MRR: reciprocal rank of first evidence hit in retrieved results
                    mrr_scores = []
                    for r in results_with_evidence:
                        ev_set = set(r.evidence_ids)
                        rr = 0.0
                        for rank, rid in enumerate(r.retrieved_dialog_ids, 1):
                            if rid in ev_set:
                                rr = 1.0 / rank
                                break
                        mrr_scores.append(rr)
                    final_results.mrr = sum(mrr_scores) / len(mrr_scores) * 100

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

            # Print results for this run
            if runs > 1:
                if retrieval_only:
                    print(f"\n  Run {run_idx + 1} R@10: {final_results.recall_at_10:.1f}%  R@20: {final_results.recall_at_20:.1f}%")
                else:
                    print(f"\n  Run {run_idx + 1} accuracy: {final_results.llm_judge_accuracy:.1f}%")
            else:
                print_results(final_results, retrieval_only=retrieval_only)

            run_accuracies.append(final_results.recall_at_10 if retrieval_only else final_results.llm_judge_accuracy)

            # Save detailed results if requested (last run's results)
            if output_path and run_idx == runs - 1:
                save_results(final_results, results, output_path)

        # Multi-run summary
        if runs > 1:
            import statistics
            mean_acc = statistics.mean(run_accuracies)
            stddev_acc = statistics.stdev(run_accuracies) if len(run_accuracies) > 1 else 0.0

            print_results(final_results, retrieval_only=retrieval_only)  # Print last run's detailed results

            print(f"\n{'=' * 70}")
            metric_name = "R@10" if retrieval_only else "accuracy"
            print(f"MULTI-RUN SUMMARY ({runs} runs)")
            print(f"{'=' * 70}")
            print(f"  Per-run {metric_name}: {', '.join(f'{a:.1f}%' for a in run_accuracies)}")
            print(f"  Mean {metric_name}:      {mean_acc:.1f}% ± {stddev_acc:.1f}%")
            print(f"{'=' * 70}")

        return final_results


def print_results(results: EvaluationResults, retrieval_only: bool = False):
    """Print formatted evaluation results"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS" + (" (retrieval-only)" if retrieval_only else ""))
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Conversations:     {results.num_conversations}")
    print(f"  Documents:         {results.num_documents}")
    print(f"  Questions:         {results.total_questions}")

    if not retrieval_only:
        print(f"\n{'='*70}")
        print("OVERALL METRICS")
        print("=" * 70)
        print(f"  Accuracy:            {results.llm_judge_accuracy:.1f}%")
        print(f"  F1 Score:            {results.avg_f1_score:.1f}%")
        print(f"  BLEU-1 Score:        {results.avg_bleu_score:.1f}%")

    print(f"\n{'='*70}")
    print("RETRIEVAL METRICS")
    print("=" * 70)
    print(f"  Recall@5:            {results.recall_at_5:.1f}%")
    print(f"  Recall@10:           {results.recall_at_10:.1f}%")
    print(f"  Recall@20:           {results.recall_at_20:.1f}%")
    print(f"  MRR:                 {results.mrr:.1f}%")
    if results.recall_at_10 > 0 and results.recall_at_20 > 0:
        if results.recall_at_10 < 50 and results.recall_at_20 > 70:
            print(f"  Diagnosis:           Ranking problem (right memories found but buried)")
        elif results.recall_at_20 < 50:
            print(f"  Diagnosis:           Recall problem (memories not found at all)")
        elif not retrieval_only and results.recall_at_10 > 70 and results.llm_judge_accuracy < 60:
            print(f"  Diagnosis:           Reasoning problem (outside MnemeFusion scope)")

    print(f"\n{'='*70}")
    print("PERFORMANCE")
    print("=" * 70)
    print(f"  Retrieval Latency:")
    print(f"    Average:           {results.avg_retrieval_latency_ms:.1f}ms")
    print(f"    P50:               {results.p50_retrieval_latency_ms:.1f}ms")
    print(f"    P95:               {results.p95_retrieval_latency_ms:.1f}ms")
    print(f"    P99:               {results.p99_retrieval_latency_ms:.1f}ms")
    if not retrieval_only:
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

    if retrieval_only:
        print(f"  {'Category':<25} {'Count':>6} {'R@5':>7} {'R@10':>7} {'R@20':>7}")
        print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    else:
        print(f"  {'Category':<25} {'Count':>6} {'Judge':>8} {'R@5':>7} {'R@10':>7} {'R@20':>7}")
        print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for cat in sorted(results.category_results.keys()):
        cat_data = results.category_results[cat]
        cat_name = category_names.get(cat, f"Category {cat}")
        r5 = f"{cat_data.get('recall_at_5', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        r10 = f"{cat_data.get('recall_at_10', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        r20 = f"{cat_data.get('recall_at_20', 0):.0f}%" if cat_data.get('questions_with_evidence', 0) > 0 else "N/A"
        if retrieval_only:
            print(f"  {cat_name:<25} {cat_data['count']:>6} {r5:>7} {r10:>7} {r20:>7}")
        else:
            print(f"  {cat_name:<25} {cat_data['count']:>6} {cat_data['llm_judge_accuracy']:>7.1f}% {r5:>7} {r10:>7} {r20:>7}")

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
    python run_eval.py

    # Quick test (50 questions)
    python run_eval.py --max-questions 50

    # With SLM metadata extraction
    python run_eval.py --use-slm --slm-model opt/models/qwen3-0.6b.gguf

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
        "--kg-model",
        default=None,
        help="Path to Triplex GGUF model for KG extraction (Full tier, requires --use-llm)"
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
        "--extraction-passes",
        type=int,
        default=1,
        help="Number of extraction passes per document (default: 1, multi-pass: 2-3)"
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
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model. Two modes: (1) HuggingFace model ID (e.g. BAAI/bge-base-en-v1.5) — "
             "uses sentence-transformers in Python; (2) local directory path (e.g. models/bge-base-en-v1.5) "
             "— uses fastembed in Rust (no sentence-transformers needed)."
    )
    parser.add_argument(
        "--nscale-extract",
        action="store_true",
        help="Use NScale cloud API for entity extraction (Qwen3-4B full-precision). "
             "Requires NSCALE_SERVICE_TOKEN env var or --nscale-token."
    )
    parser.add_argument(
        "--nscale-token",
        default=None,
        help="NScale API service token (overrides NSCALE_SERVICE_TOKEN env var)"
    )
    parser.add_argument(
        "--nscale-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="NScale model for extraction (default: Qwen/Qwen3-4B-Instruct-2507)"
    )
    parser.add_argument(
        "--session-expand",
        action="store_true",
        help="Enable session-level context expansion. When multiple retrieved turns "
             "come from the same session, expand to include neighboring turns. "
             "Replaces lowest-ranked singletons to stay within top_k budget."
    )
    parser.add_argument(
        "--neighbor-expand",
        action="store_true",
        help="Enable turn neighborhood expansion. For each retrieved turn, add ±2 "
             "neighboring turns from the same session. Addresses near-miss retrieval "
             "where the right session is found but the wrong turn is selected."
    )
    parser.add_argument(
        "--two-step-rag",
        action="store_true",
        help="Enable two-step RAG: (1) extract relevant facts from context, "
             "(2) answer from extracted facts. Research: Chain-of-Note +10pts. "
             "Adds ~$0.001/query cost (GPT-4o-mini)."
    )
    parser.add_argument(
        "--mcq",
        action="store_true",
        help="MCQ evaluation mode: use 10-choice multiple-choice questions from "
             "Percena/locomo-mc10 instead of free-text generation + LLM judge. "
             "Deterministic, no judge noise, ~50%% cheaper. Based on Percena/locomo-mc10."
    )
    parser.add_argument(
        "--mcq-path",
        default=None,
        help="Path to locomo_mc10.json JSONL file. Auto-detected from HuggingFace cache if not specified."
    )
    parser.add_argument(
        "--adaptive-k",
        type=float,
        default=0.0,
        help="Adaptive-K (Top-p) threshold for dynamic result count. "
             "0.0=disabled (default), 0.7=recommended. Reduces context when "
             "low-quality results would dilute it. Based on Percena/locomo-mc10."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs. For publication, use 3-5 runs to report "
             "mean ± stddev. Default: 1."
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retrieval-only mode: skip LLM answer generation and judging. "
             "Reports R@5, R@10, R@20, and MRR. No OpenAI API key needed."
    )

    args = parser.parse_args()
    # Check API key (not needed for retrieval-only mode)
    if not args.retrieval_only and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Resolve NScale token
    nscale_token = args.nscale_token or os.environ.get("NSCALE_SERVICE_TOKEN")
    if args.nscale_extract and not nscale_token:
        print("ERROR: --nscale-extract requires NSCALE_SERVICE_TOKEN env var or --nscale-token")
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
        kg_model_path=args.kg_model,
        output_path=args.output,
        verbose=args.verbose,
        db_path=args.db_path,
        skip_ingestion=args.skip_ingestion,
        extraction_passes=args.extraction_passes,
        embedding_model=args.embedding_model,
        nscale_extract=args.nscale_extract,
        nscale_token=nscale_token,
        nscale_model=args.nscale_model,
        session_expand=args.session_expand,
        neighbor_expand=args.neighbor_expand,
        two_step_rag=args.two_step_rag,
        mcq=args.mcq,
        mcq_path=args.mcq_path,
        adaptive_k_threshold=args.adaptive_k,
        runs=args.runs,
        retrieval_only=args.retrieval_only,
    )


if __name__ == "__main__":
    main()
