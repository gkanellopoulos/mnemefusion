#!/usr/bin/env python3
"""
LoCoMo Benchmark Evaluation for MnemeFusion

This script evaluates MnemeFusion's retrieval performance on the LoCoMo dataset,
a long-term conversational memory benchmark.

Dataset: https://github.com/snap-research/locomo
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"

Usage:
    python locomo_eval.py --samples 1   # Phase 1: Validation (1 conversation)
    python locomo_eval.py --samples 10  # Phase 2: Full evaluation (all conversations)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile

# Add parent directory to path for mnemefusion imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemefusion-python"))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install sentence-transformers numpy")
    sys.exit(1)


class LoCoMoEvaluator:
    """Evaluates MnemeFusion on LoCoMo dataset"""

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

    def load_dataset(self, dataset_path: str, num_conversations: int = None) -> List[Dict]:
        """
        Load LoCoMo dataset from JSON file

        Args:
            dataset_path: Path to locomo10.json
            num_conversations: Number of conversations to load (None = all)

        Returns:
            List of conversation samples
        """
        print(f"\nLoading LoCoMo dataset from {dataset_path}...")

        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found at {dataset_path}")
            print("\nTo download the dataset:")
            print("  1. Clone: git clone https://github.com/snap-research/locomo.git")
            print("  2. Copy: cp locomo/data/locomo10.json tests/benchmarks/fixtures/")
            sys.exit(1)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = data if isinstance(data, list) else [data]

        if num_conversations:
            conversations = conversations[:num_conversations]

        print(f"[OK] Loaded {len(conversations)} conversations")
        return conversations

    def prepare_documents(self, conversations: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """
        Prepare conversation turns as documents

        Args:
            conversations: List of LoCoMo conversation samples

        Returns:
            List of (doc_id, content, metadata) tuples
        """
        print("\nPreparing conversation turns as documents...")
        documents = []

        for conv_idx, conversation in enumerate(conversations):
            sample_id = conversation.get('sample_id', f'conv_{conv_idx}')
            conv_data = conversation.get('conversation', {})

            # Extract speaker names
            speaker_a = conv_data.get('speaker_a', 'Speaker_A')
            speaker_b = conv_data.get('speaker_b', 'Speaker_B')

            # Iterate through sessions (session_1, session_2, etc.)
            session_idx = 1
            while f'session_{session_idx}' in conv_data:
                session_key = f'session_{session_idx}'
                session_date_key = f'session_{session_idx}_date_time'

                turns = conv_data.get(session_key, [])
                session_date = conv_data.get(session_date_key, f'session_{session_idx}')

                for turn_idx, turn in enumerate(turns):
                    speaker = turn.get('speaker', 'unknown')
                    text = turn.get('text', '')
                    dialog_id = turn.get('dia_id', f'{sample_id}_s{session_idx}_t{turn_idx}')

                    if not text:
                        continue

                    # Create document ID (use dia_id from dataset)
                    doc_id = dialog_id

                    # Metadata
                    metadata = {
                        'conversation_id': sample_id,
                        'session_idx': str(session_idx),
                        'session_date': session_date,
                        'turn_idx': str(turn_idx),
                        'speaker': speaker,
                        'dialog_id': dialog_id
                    }

                    documents.append((doc_id, text, metadata))

                session_idx += 1

        print(f"[OK] Prepared {len(documents)} conversation turns from {len(conversations)} conversations")
        return documents

    def extract_qa_pairs(self, conversations: List[Dict]) -> List[Dict]:
        """
        Extract question-answer pairs from conversations

        Args:
            conversations: List of LoCoMo conversation samples

        Returns:
            List of QA dictionaries with question, answer, evidence_dialog_ids, category
        """
        print("\nExtracting QA pairs...")
        qa_pairs = []

        for conversation in conversations:
            sample_id = conversation.get('sample_id', 'unknown')
            qa_list = conversation.get('qa', [])

            for qa in qa_list:
                qa_pairs.append({
                    'conversation_id': sample_id,
                    'question': qa.get('question', ''),
                    'answer': qa.get('answer', ''),
                    'evidence_dialog_ids': qa.get('evidence', []),  # Dataset uses 'evidence' key
                    'category': qa.get('category', 'unknown')
                })

        print(f"[OK] Extracted {len(qa_pairs)} QA pairs")
        return qa_pairs

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
        Build MnemeFusion database with conversation turns

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
        print("Generating embeddings for conversation turns...")
        contents = [doc[1] for doc in documents]
        embeddings = self.generate_embeddings(contents)

        # Add documents to database
        print("Adding conversation turns to database...")
        for i, ((doc_id, content, metadata), embedding) in enumerate(zip(documents, embeddings)):
            # Convert metadata to string dict (Python bindings requirement)
            metadata_str = {k: str(v) for k, v in metadata.items()}

            engine.add(
                content=content,
                embedding=embedding.tolist(),
                metadata=metadata_str,
                timestamp=None,
                source=None,
                namespace=None
            )

            if (i + 1) % 100 == 0:
                print(f"  Added {i + 1}/{len(documents)} turns")

        print(f"[OK] Database built with {len(documents)} conversation turns")
        return engine

    def evaluate(self, qa_pairs: List[Dict], engine, top_k: int = 10) -> Dict:
        """
        Evaluate retrieval performance on QA pairs

        Args:
            qa_pairs: List of QA dictionaries
            engine: MnemeFusion engine
            top_k: Number of top results to retrieve

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating retrieval (top-{top_k})...")

        total_queries = len(qa_pairs)
        recall_at_k_scores = []
        mrr_scores = []
        precision_at_k_scores = []

        # Track by category
        category_metrics = {}

        # BGE instruction for queries
        query_instruction = "Represent this sentence for searching relevant passages: "

        for i, qa in enumerate(qa_pairs):
            # Generate query embedding
            query = qa['question']
            query_embedding = self.generate_embeddings([query], instruction=query_instruction)[0]

            # Search in MnemeFusion
            results = engine.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k,
                namespace=None,
                filters=None
            )

            # Get evidence dialog IDs (ground truth)
            evidence_ids = set(qa['evidence_dialog_ids'])

            # Check which retrieved documents are relevant
            retrieved_ids = []
            for memory, score in results:
                # Extract dialog_id from metadata
                dialog_id = memory['metadata'].get('dialog_id', '')
                retrieved_ids.append(dialog_id)

            # Calculate metrics
            relevant_retrieved = [id for id in retrieved_ids if id in evidence_ids]

            # Recall@K: How many evidence turns were retrieved?
            recall = len(relevant_retrieved) / len(evidence_ids) if evidence_ids else 0
            recall_at_k_scores.append(recall)

            # MRR: Reciprocal rank of first relevant document
            first_relevant_rank = None
            for rank, id in enumerate(retrieved_ids, 1):
                if id in evidence_ids:
                    first_relevant_rank = rank
                    break
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            mrr_scores.append(mrr)

            # Precision@K: Fraction of retrieved that are relevant
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            precision_at_k_scores.append(precision)

            # Track by category
            category = qa['category']
            if category not in category_metrics:
                category_metrics[category] = {'recall': [], 'mrr': [], 'precision': []}
            category_metrics[category]['recall'].append(recall)
            category_metrics[category]['mrr'].append(mrr)
            category_metrics[category]['precision'].append(precision)

            if (i + 1) % 10 == 0 or (i + 1) == total_queries:
                print(f"  Evaluated {i + 1}/{total_queries} queries")

        # Aggregate metrics
        metrics = {
            'num_queries': total_queries,
            'recall_at_k': np.mean(recall_at_k_scores),
            'mrr': np.mean(mrr_scores),
            'precision_at_k': np.mean(precision_at_k_scores),
            'top_k': top_k,
            'category_metrics': {}
        }

        # Add per-category metrics
        for category, scores in category_metrics.items():
            metrics['category_metrics'][category] = {
                'recall_at_k': np.mean(scores['recall']),
                'mrr': np.mean(scores['mrr']),
                'precision_at_k': np.mean(scores['precision']),
                'count': len(scores['recall'])
            }

        return metrics

    def print_results(self, metrics: Dict, phase: str = "1"):
        """Print evaluation results"""
        print("\n" + "="*60)
        print(f"LoCoMo Evaluation Results - Phase {phase}")
        print("="*60)
        print(f"Queries evaluated:    {metrics['num_queries']}")
        print(f"Top-K retrieved:      {metrics['top_k']}")
        print(f"\nOverall Metrics:")
        print(f"  Recall@{metrics['top_k']:>2}:         {metrics['recall_at_k']:.3f} ({metrics['recall_at_k']*100:.1f}%)")
        print(f"  MRR:                {metrics['mrr']:.3f}")
        print(f"  Precision@{metrics['top_k']:>2}:      {metrics['precision_at_k']:.3f} ({metrics['precision_at_k']*100:.1f}%)")

        if metrics.get('category_metrics'):
            print(f"\nMetrics by Category:")
            for category, scores in metrics['category_metrics'].items():
                print(f"  {category} ({scores['count']} queries):")
                print(f"    Recall@{metrics['top_k']}: {scores['recall_at_k']:.3f}")
                print(f"    MRR:        {scores['mrr']:.3f}")

        print("\n" + "="*60)

        if phase == "2":
            target_recall = 0.70
            if metrics['recall_at_k'] >= target_recall:
                print(f"[SUCCESS] Recall@10 ({metrics['recall_at_k']:.3f}) >= target ({target_recall:.3f})")
            else:
                print(f"[BELOW TARGET] Recall@10 ({metrics['recall_at_k']:.3f}) < target ({target_recall:.3f})")
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
    parser = argparse.ArgumentParser(description='LoCoMo Benchmark Evaluation')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of conversations to evaluate (1 for Phase 1, 10 for Phase 2)')
    parser.add_argument('--dataset', type=str,
                       default='tests/benchmarks/fixtures/locomo10.json',
                       help='Path to locomo10.json dataset')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top results to retrieve')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Determine phase
    phase = "1" if args.samples <= 1 else "2"

    print("="*60)
    print(f"LoCoMo Benchmark Evaluation - Phase {phase}")
    print("="*60)
    print(f"Conversations: {args.samples}")
    print(f"Top-K: {args.top_k}")
    print(f"GPU: {'Enabled' if not args.no_gpu else 'Disabled'}")
    print()

    # Create evaluator
    evaluator = LoCoMoEvaluator(use_gpu=not args.no_gpu)

    # Load dataset
    conversations = evaluator.load_dataset(args.dataset, num_conversations=args.samples)

    # Extract QA pairs
    qa_pairs = evaluator.extract_qa_pairs(conversations)

    if not qa_pairs:
        print("ERROR: No QA pairs found in dataset")
        sys.exit(1)

    # Prepare documents (conversation turns)
    documents = evaluator.prepare_documents(conversations)

    # Build database in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "locomo_eval.mfdb")

        # Build MnemeFusion database
        engine = evaluator.build_memory_database(documents, db_path)

        # Evaluate
        start_time = time.time()
        metrics = evaluator.evaluate(qa_pairs, engine, top_k=args.top_k)
        elapsed = time.time() - start_time

        metrics['elapsed_seconds'] = elapsed
        metrics['phase'] = phase
        metrics['model'] = "BAAI/bge-base-en-v1.5"
        metrics['embedding_dim'] = evaluator.embedding_dim
        metrics['num_conversations'] = len(conversations)

    # Print results
    evaluator.print_results(metrics, phase=phase)
    print(f"Evaluation time: {elapsed:.1f} seconds")

    # Save results
    if args.output:
        evaluator.save_results(metrics, args.output)
    else:
        # Default output path
        output_path = f"tests/benchmarks/fixtures/locomo_phase{phase}_results.json"
        evaluator.save_results(metrics, output_path)


if __name__ == '__main__':
    main()
