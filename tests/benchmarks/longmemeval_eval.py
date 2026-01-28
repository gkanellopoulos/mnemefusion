#!/usr/bin/env python3
"""
LongMemEval Benchmark Evaluation for MnemeFusion

This script evaluates MnemeFusion's retrieval performance on the LongMemEval dataset,
testing 5 core memory abilities: information extraction, multi-session reasoning,
temporal reasoning, knowledge updates, and abstention.

Dataset: https://github.com/xiaowu0162/LongMemEval
Paper: "Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?"

Usage:
    python longmemeval_eval.py --variant s        # LongMemEval_S (~115k tokens, for 128k models)
    python longmemeval_eval.py --variant oracle   # LongMemEval_Oracle (evidence sessions only)
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Add parent directory to path for mnemefusion imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemefusion-python"))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install sentence-transformers numpy")
    sys.exit(1)


class LongMemEvalEvaluator:
    """Evaluates MnemeFusion on LongMemEval dataset"""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", use_gpu: bool = True):
        """
        Initialize evaluator with embedding model

        Args:
            model_name: SentenceTransformer model name
            use_gpu: Whether to use GPU for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
                    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                else:
                    print("[WARNING] GPU requested but not available, using CPU")
            except ImportError:
                print("[WARNING] PyTorch not found, using CPU")
        else:
            print("Using CPU for embeddings")

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def download_dataset(self, variant: str, data_dir: str = "tests/benchmarks/fixtures") -> str:
        """
        Download LongMemEval dataset if not already present

        Args:
            variant: Dataset variant ('s', 'm', or 'oracle')
            data_dir: Directory to save dataset

        Returns:
            Path to downloaded dataset
        """
        os.makedirs(data_dir, exist_ok=True)

        variant_map = {
            's': 'longmemeval_s_cleaned.json',
            'm': 'longmemeval_m_cleaned.json',
            'oracle': 'longmemeval_oracle.json'
        }

        if variant not in variant_map:
            print(f"ERROR: Invalid variant '{variant}'. Choose from: s, m, oracle")
            sys.exit(1)

        filename = variant_map[variant]
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            print(f"[OK] Dataset already exists: {filepath}")
            return filepath

        print(f"\nDownloading LongMemEval variant '{variant}'...")
        base_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
        url = f"{base_url}/{filename}"

        try:
            import urllib.request
            print(f"Fetching from: {url}")
            urllib.request.urlretrieve(url, filepath)
            print(f"[OK] Downloaded to: {filepath}")
            return filepath
        except Exception as e:
            print(f"ERROR: Failed to download dataset: {e}")
            print("\nManual download instructions:")
            print(f"  wget {url} -O {filepath}")
            sys.exit(1)

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load LongMemEval dataset from JSON file

        Args:
            dataset_path: Path to longmemeval JSON file

        Returns:
            List of evaluation instances
        """
        print(f"\nLoading LongMemEval dataset from {dataset_path}...")

        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found at {dataset_path}")
            sys.exit(1)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = data if isinstance(data, list) else [data]
        print(f"[OK] Loaded {len(instances)} evaluation instances")

        return instances

    def prepare_documents(self, instances: List[Dict]) -> Tuple[List[Tuple[str, str, Dict]], int]:
        """
        Prepare haystack sessions as documents

        Args:
            instances: List of LongMemEval evaluation instances

        Returns:
            Tuple of (documents list, total token count estimate)
        """
        print("\nPreparing haystack sessions as documents...")
        documents = []
        total_tokens = 0

        for instance_idx, instance in enumerate(instances):
            question_id = instance.get('question_id', f'q_{instance_idx}')
            haystack_sessions = instance.get('haystack_sessions', [])
            haystack_dates = instance.get('haystack_dates', [])
            haystack_session_ids = instance.get('haystack_session_ids', [])

            for session_idx, session in enumerate(haystack_sessions):
                # Each session is a conversation with turns
                session_id = haystack_session_ids[session_idx] if session_idx < len(haystack_session_ids) else f'session_{session_idx}'
                session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else None

                # Concatenate all turns in session
                session_text = ""
                for turn in session:
                    role = turn.get('role', 'user')
                    content = turn.get('content', '')
                    session_text += f"{role}: {content}\n"

                # Estimate tokens (rough: 1 token ≈ 4 chars)
                total_tokens += len(session_text) // 4

                # Create document ID
                doc_id = f"{question_id}_session_{session_id}"

                # Metadata
                metadata = {
                    'question_id': question_id,
                    'session_id': str(session_id),
                    'session_date': str(session_date) if session_date else '',
                    'session_idx': str(session_idx),
                }

                documents.append((doc_id, session_text.strip(), metadata))

        print(f"[OK] Prepared {len(documents)} sessions (~{total_tokens:,} tokens)")
        return documents, total_tokens

    def generate_embeddings(self, texts: List[str], instruction: str = None) -> np.ndarray:
        """
        Generate embeddings for texts using BGE model

        Args:
            texts: List of texts to embed
            instruction: Optional instruction prefix (for queries)

        Returns:
            Numpy array of embeddings
        """
        if instruction:
            texts = [instruction + text for text in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # Cosine similarity
            show_progress_bar=len(texts) > 10,
            batch_size=32
        )

        return embeddings

    def build_memory_database(self, documents: List[Tuple[str, str, Dict]], db_path: str):
        """
        Build MnemeFusion database with haystack sessions

        Args:
            documents: List of (doc_id, content, metadata) tuples
            db_path: Path to save database
        """
        print(f"\nBuilding MnemeFusion database at {db_path}...")

        # Import MnemeFusion Python bindings
        try:
            import mnemefusion
        except ImportError:
            print("ERROR: mnemefusion Python package not installed")
            print("Install with: cd mnemefusion-python && maturin develop")
            sys.exit(1)

        # Create config dict with correct embedding dimension
        config = {"embedding_dim": self.embedding_dim}

        # Open/create database
        engine = mnemefusion.Memory(db_path, config=config)

        # Reserve capacity for all documents (improves performance)
        print(f"Reserving capacity for {len(documents)} documents...")
        engine.reserve_capacity(len(documents))

        # Generate embeddings for all documents
        print("Generating embeddings for sessions...")
        contents = [doc[1] for doc in documents]
        embeddings = self.generate_embeddings(contents)

        # Add documents to database
        print("Adding sessions to database...")
        for i, ((doc_id, content, metadata), embedding) in enumerate(zip(documents, embeddings)):
            # Convert metadata to string dict (Python bindings requirement)
            metadata_str = {k: str(v) for k, v in metadata.items()}

            # Parse session_date for timestamp if available
            timestamp = None
            if metadata.get('session_date'):
                try:
                    from datetime import datetime
                    # Try to parse date (format depends on dataset)
                    # LongMemEval uses formats like "2023-01-15"
                    date_str = metadata['session_date']
                    if date_str and date_str != 'None':
                        dt = datetime.fromisoformat(date_str)
                        timestamp = int(dt.timestamp())
                except Exception:
                    pass  # Skip if parsing fails

            engine.add(
                content=content,
                embedding=embedding.tolist(),
                metadata=metadata_str,
                timestamp=timestamp,
                source=None,
                namespace=None
            )

            if (i + 1) % 100 == 0:
                print(f"  Added {i + 1}/{len(documents)} sessions")

        print(f"[OK] Database built with {len(documents)} sessions")
        return engine

    def evaluate(self, instances: List[Dict], engine, top_k: int = 10) -> Dict:
        """
        Evaluate retrieval performance on LongMemEval instances

        Args:
            instances: List of LongMemEval evaluation instances
            engine: MnemeFusion engine
            top_k: Number of top results to retrieve

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating retrieval (top-{top_k})...")

        total_questions = len(instances)
        recall_scores = []
        precision_scores = []
        f1_scores = []

        # Track by question type and ability
        type_metrics = defaultdict(lambda: {'recall': [], 'precision': [], 'f1': []})
        ability_metrics = defaultdict(lambda: {'recall': [], 'precision': [], 'f1': []})

        # Map question types to abilities
        type_to_ability = {
            'single-session-user': 'Information Extraction',
            'single-session-assistant': 'Information Extraction',
            'single-session-preference': 'Information Extraction',
            'temporal-reasoning': 'Temporal Reasoning',
            'knowledge-update': 'Knowledge Updates',
            'multi-session': 'Multi-Session Reasoning',
        }

        # BGE instruction for queries
        query_instruction = "Represent this sentence for searching relevant passages: "

        for i, instance in enumerate(instances):
            question_id = instance.get('question_id', f'q_{i}')
            question = instance.get('question', '')
            question_type = instance.get('question_type', 'unknown')
            answer_session_ids = set(str(sid) for sid in instance.get('answer_session_ids', []))

            # Determine if this is abstention query
            is_abstention = question_type.endswith('_abs')
            ability = 'Abstention' if is_abstention else type_to_ability.get(
                question_type.replace('_abs', ''), 'Unknown'
            )

            # Generate query embedding
            query_embedding = self.generate_embeddings([question], instruction=query_instruction)[0]

            # Use 4D fusion query
            intent, results = engine.query(
                query_text=question,
                query_embedding=query_embedding.tolist(),
                limit=top_k,
                namespace=None,
                filters=None
            )

            # Extract retrieved session IDs
            retrieved_session_ids = set()
            for memory, scores in results:
                session_id = memory['metadata'].get('session_id', '')
                if session_id:
                    retrieved_session_ids.add(session_id)

            # Calculate metrics
            if answer_session_ids:
                # True positives: sessions that should be retrieved and were retrieved
                true_positives = len(retrieved_session_ids & answer_session_ids)

                # Recall: fraction of relevant sessions retrieved
                recall = true_positives / len(answer_session_ids)
                recall_scores.append(recall)

                # Precision: fraction of retrieved sessions that are relevant
                precision = true_positives / len(retrieved_session_ids) if retrieved_session_ids else 0.0
                precision_scores.append(precision)

                # F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)

                # Track by type and ability
                type_metrics[question_type]['recall'].append(recall)
                type_metrics[question_type]['precision'].append(precision)
                type_metrics[question_type]['f1'].append(f1)

                ability_metrics[ability]['recall'].append(recall)
                ability_metrics[ability]['precision'].append(precision)
                ability_metrics[ability]['f1'].append(f1)

            if (i + 1) % 10 == 0 or (i + 1) == total_questions:
                print(f"  Evaluated {i + 1}/{total_questions} questions")

        # Aggregate metrics
        metrics = {
            'num_questions': total_questions,
            'recall': np.mean(recall_scores) if recall_scores else 0.0,
            'precision': np.mean(precision_scores) if precision_scores else 0.0,
            'f1': np.mean(f1_scores) if f1_scores else 0.0,
            'top_k': top_k,
            'type_metrics': {},
            'ability_metrics': {}
        }

        # Add per-type metrics
        for qtype, scores in type_metrics.items():
            metrics['type_metrics'][qtype] = {
                'recall': np.mean(scores['recall']),
                'precision': np.mean(scores['precision']),
                'f1': np.mean(scores['f1']),
                'count': len(scores['recall'])
            }

        # Add per-ability metrics
        for ability, scores in ability_metrics.items():
            metrics['ability_metrics'][ability] = {
                'recall': np.mean(scores['recall']),
                'precision': np.mean(scores['precision']),
                'f1': np.mean(scores['f1']),
                'count': len(scores['recall'])
            }

        return metrics

    def print_results(self, metrics: Dict, variant: str):
        """Print evaluation results"""
        print("\n" + "="*60)
        print(f"LongMemEval Evaluation Results - Variant: {variant.upper()}")
        print("="*60)
        print(f"Questions evaluated:  {metrics['num_questions']}")
        print(f"Top-K retrieved:      {metrics['top_k']}")
        print(f"\nOverall Metrics:")
        print(f"  Recall:      {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"  Precision:   {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"  F1 Score:    {metrics['f1']:.3f} ({metrics['f1']*100:.1f}%)")

        if metrics.get('ability_metrics'):
            print(f"\nMetrics by Memory Ability:")
            abilities_order = [
                'Information Extraction',
                'Multi-Session Reasoning',
                'Temporal Reasoning',
                'Knowledge Updates',
                'Abstention'
            ]
            for ability in abilities_order:
                if ability in metrics['ability_metrics']:
                    scores = metrics['ability_metrics'][ability]
                    print(f"  {ability} ({scores['count']} questions):")
                    print(f"    Recall:    {scores['recall']:.3f}")
                    print(f"    Precision: {scores['precision']:.3f}")
                    print(f"    F1:        {scores['f1']:.3f}")

        if metrics.get('type_metrics'):
            print(f"\nMetrics by Question Type:")
            for qtype, scores in sorted(metrics['type_metrics'].items()):
                print(f"  {qtype} ({scores['count']} questions):")
                print(f"    Recall: {scores['recall']:.3f}, Precision: {scores['precision']:.3f}, F1: {scores['f1']:.3f}")

        print("\n" + "="*60)

        # Compare to target
        target = 0.70
        if metrics['f1'] >= target:
            print(f"[SUCCESS] F1 Score ({metrics['f1']:.3f}) >= target ({target:.3f})")
        else:
            print(f"[BELOW TARGET] F1 Score ({metrics['f1']:.3f}) < target ({target:.3f})")
        print("="*60)

    def save_results(self, metrics: Dict, output_path: str):
        """Save results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        metrics_native = convert_to_native(metrics)

        with open(output_path, 'w') as f:
            json.dump(metrics_native, f, indent=2)
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LongMemEval Benchmark Evaluation')
    parser.add_argument('--variant', type=str, default='s', choices=['s', 'm', 'oracle'],
                       help="Dataset variant: 's' (~115k tokens), 'm' (~500 sessions), 'oracle' (evidence only)")
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to LongMemEval JSON file (auto-downloads if not specified)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top results to retrieve')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    print("="*60)
    print(f"LongMemEval Benchmark Evaluation - Variant: {args.variant.upper()}")
    print("="*60)
    print(f"Top-K: {args.top_k}")
    print(f"GPU: {'Enabled' if not args.no_gpu else 'Disabled'}")
    print()

    # Create evaluator
    evaluator = LongMemEvalEvaluator(use_gpu=not args.no_gpu)

    # Download or load dataset
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = evaluator.download_dataset(args.variant)

    # Load dataset
    instances = evaluator.load_dataset(dataset_path)

    if not instances:
        print("ERROR: No evaluation instances found in dataset")
        sys.exit(1)

    # Prepare documents (haystack sessions)
    documents, total_tokens = evaluator.prepare_documents(instances)

    # Build database in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "longmemeval.mfdb")

        # Build MnemeFusion database
        engine = evaluator.build_memory_database(documents, db_path)

        # Evaluate
        start_time = time.time()
        metrics = evaluator.evaluate(instances, engine, top_k=args.top_k)
        elapsed = time.time() - start_time

        metrics['elapsed_seconds'] = elapsed
        metrics['variant'] = args.variant
        metrics['model'] = "BAAI/bge-base-en-v1.5"
        metrics['embedding_dim'] = evaluator.embedding_dim
        metrics['total_sessions'] = len(documents)
        metrics['total_tokens_estimate'] = total_tokens

    # Print results
    evaluator.print_results(metrics, args.variant)
    print(f"Evaluation time: {elapsed:.1f} seconds")

    # Save results
    if args.output:
        evaluator.save_results(metrics, args.output)
    else:
        # Default output path
        output_path = f"tests/benchmarks/fixtures/longmemeval_{args.variant}_results.json"
        evaluator.save_results(metrics, output_path)


if __name__ == '__main__':
    main()
