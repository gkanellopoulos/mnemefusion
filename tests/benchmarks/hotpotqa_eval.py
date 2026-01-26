#!/usr/bin/env python3
"""
HotpotQA Benchmark Evaluation for MnemeFusion

This script evaluates MnemeFusion's retrieval performance on the HotpotQA dataset,
a multi-hop question answering benchmark.

Dataset: https://hotpotqa.github.io/
Paper: "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"

Usage:
    python hotpotqa_eval.py --samples 10   # Phase 1: Validation (10 samples)
    python hotpotqa_eval.py --samples 1000 # Phase 2: Full evaluation
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
    print("Install with: pip install sentence-transformers numpy datasets")
    sys.exit(1)


class HotpotQAEvaluator:
    """Evaluates MnemeFusion on HotpotQA dataset"""

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

    def download_dataset(self, num_samples: int = 10) -> List[Dict]:
        """
        Download HotpotQA dataset

        Args:
            num_samples: Number of samples to download (10 for Phase 1, 1000 for Phase 2)

        Returns:
            List of dataset samples
        """
        print(f"\nDownloading HotpotQA dataset ({num_samples} samples)...")

        try:
            from datasets import load_dataset

            # Load HotpotQA from Hugging Face datasets
            # Using 'distractor' setting which includes both supporting and distractor passages
            dataset = load_dataset('hotpot_qa', 'distractor', split='validation')

            # Take requested number of samples
            samples = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break

                # HotpotQA format:
                # - question: The question to answer
                # - context: List of [title, sentences] pairs (supporting + distractors)
                # - supporting_facts: List of [title, sentence_idx] pairs (ground truth)
                # - answer: The correct answer

                samples.append({
                    'id': item['id'],
                    'question': item['question'],
                    'context': item['context']['title'],  # All passage titles
                    'sentences': item['context']['sentences'],  # All sentences per passage
                    'supporting_facts': item['supporting_facts'],  # Ground truth
                    'answer': item['answer'],
                    'type': item['type']  # 'bridge' or 'comparison'
                })

            print(f"[OK] Downloaded {len(samples)} samples")
            return samples

        except ImportError:
            print("ERROR: 'datasets' package not installed")
            print("Install with: pip install datasets")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR downloading dataset: {e}")
            sys.exit(1)

    def prepare_documents(self, samples: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """
        Prepare documents (passages) from HotpotQA samples

        Args:
            samples: List of HotpotQA samples

        Returns:
            List of (doc_id, content, metadata) tuples
        """
        print("\nPreparing documents...")
        documents = []

        for sample in samples:
            # Each sample has multiple context passages (titles + sentences)
            for title, sentences in zip(sample['context'], sample['sentences']):
                # Concatenate sentences into single passage
                content = ' '.join(sentences)

                # Create document ID
                doc_id = f"{sample['id']}_{title}"

                # Metadata
                metadata = {
                    'question_id': sample['id'],
                    'title': title,
                    'question': sample['question']
                }

                documents.append((doc_id, content, metadata))

        print(f"[OK] Prepared {len(documents)} documents from {len(samples)} questions")
        return documents

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
        Build MnemeFusion database with documents

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
        print("Generating embeddings for documents...")
        contents = [doc[1] for doc in documents]
        embeddings = self.generate_embeddings(contents)

        # Add documents to database
        print("Adding documents to database...")
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
                print(f"  Added {i + 1}/{len(documents)} documents")

        print(f"[OK] Database built with {len(documents)} documents")
        return engine

    def evaluate(self, samples: List[Dict], engine, top_k: int = 10) -> Dict:
        """
        Evaluate retrieval performance on samples

        Args:
            samples: List of HotpotQA samples
            engine: MnemeFusion engine
            top_k: Number of top results to retrieve

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating retrieval (top-{top_k})...")

        total_queries = len(samples)
        recall_at_k_scores = []
        mrr_scores = []
        precision_at_k_scores = []

        # Track intent distribution
        intent_counts = {}

        # BGE instruction for queries
        query_instruction = "Represent this sentence for searching relevant passages: "

        for i, sample in enumerate(samples):
            # Generate query embedding
            query = sample['question']
            query_embedding = self.generate_embeddings([query], instruction=query_instruction)[0]

            # Use 4D fusion query instead of semantic-only search
            intent, results = engine.query(
                query_text=query,  # Natural language question for intent classification
                query_embedding=query_embedding.tolist(),
                limit=top_k,
                namespace=None,
                filters=None
            )

            # Track intent distribution
            intent_type = intent['intent']
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1

            # Get supporting facts (ground truth)
            # HotpotQA format: supporting_facts = {'title': [list of titles], 'sent_id': [list of sent_ids]}
            supporting_facts = sample['supporting_facts']
            supporting_titles = set(supporting_facts['title']) if isinstance(supporting_facts, dict) else set()

            # Check which retrieved documents are relevant
            retrieved_titles = []
            for memory, scores in results:  # Changed: scores is now dict, not float
                # Extract title from metadata (memory is a dict)
                title = memory['metadata'].get('title', '')
                retrieved_titles.append(title)
                # scores['fused_score'] is the final weighted score

            # Calculate metrics
            relevant_retrieved = [title for title in retrieved_titles if title in supporting_titles]

            # Recall@K: How many supporting facts were retrieved?
            recall = len(relevant_retrieved) / len(supporting_titles) if supporting_titles else 0
            recall_at_k_scores.append(recall)

            # MRR: Reciprocal rank of first relevant document
            first_relevant_rank = None
            for rank, title in enumerate(retrieved_titles, 1):
                if title in supporting_titles:
                    first_relevant_rank = rank
                    break
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            mrr_scores.append(mrr)

            # Precision@K: Fraction of retrieved that are relevant
            precision = len(relevant_retrieved) / len(retrieved_titles) if retrieved_titles else 0
            precision_at_k_scores.append(precision)

            if (i + 1) % 10 == 0 or (i + 1) == total_queries:
                print(f"  Evaluated {i + 1}/{total_queries} queries")

        # Aggregate metrics
        metrics = {
            'num_queries': total_queries,
            'recall_at_k': np.mean(recall_at_k_scores),
            'mrr': np.mean(mrr_scores),
            'precision_at_k': np.mean(precision_at_k_scores),
            'top_k': top_k,
            'intent_distribution': {
                intent: count / total_queries for intent, count in intent_counts.items()
            }
        }

        return metrics

    def print_results(self, metrics: Dict, phase: str = "1"):
        """Print evaluation results"""
        print("\n" + "="*60)
        print(f"HotpotQA Evaluation Results - Phase {phase}")
        print("="*60)
        print(f"Queries evaluated:    {metrics['num_queries']}")
        print(f"Top-K retrieved:      {metrics['top_k']}")
        print(f"\nMetrics:")
        print(f"  Recall@{metrics['top_k']:>2}:         {metrics['recall_at_k']:.3f} ({metrics['recall_at_k']*100:.1f}%)")
        print(f"  MRR:                {metrics['mrr']:.3f}")
        print(f"  Precision@{metrics['top_k']:>2}:      {metrics['precision_at_k']:.3f} ({metrics['precision_at_k']*100:.1f}%)")

        # Print intent distribution
        if metrics.get('intent_distribution'):
            print(f"\nIntent Distribution:")
            for intent, pct in sorted(metrics['intent_distribution'].items(), key=lambda x: -x[1]):
                print(f"  {intent}: {pct*100:.1f}%")

        print("\n" + "="*60)

        if phase == "2":
            target_recall = 0.60
            if metrics['recall_at_k'] >= target_recall:
                print(f"[SUCCESS] Recall@10 ({metrics['recall_at_k']:.3f}) >= target ({target_recall:.3f})")
            else:
                print(f"[BELOW TARGET] Recall@10 ({metrics['recall_at_k']:.3f}) < target ({target_recall:.3f})")
            print("="*60)

    def save_results(self, metrics: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='HotpotQA Benchmark Evaluation')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to evaluate (10 for Phase 1, 1000 for Phase 2)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top results to retrieve')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Determine phase
    phase = "1" if args.samples <= 50 else "2"

    print("="*60)
    print(f"HotpotQA Benchmark Evaluation - Phase {phase}")
    print("="*60)
    print(f"Samples: {args.samples}")
    print(f"Top-K: {args.top_k}")
    print(f"GPU: {'Enabled' if not args.no_gpu else 'Disabled'}")
    print()

    # Create evaluator
    evaluator = HotpotQAEvaluator(use_gpu=not args.no_gpu)

    # Download dataset
    samples = evaluator.download_dataset(num_samples=args.samples)

    # Prepare documents
    documents = evaluator.prepare_documents(samples)

    # Build database in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "hotpotqa_eval.mfdb")

        # Build MnemeFusion database
        engine = evaluator.build_memory_database(documents, db_path)

        # Evaluate
        start_time = time.time()
        metrics = evaluator.evaluate(samples, engine, top_k=args.top_k)
        elapsed = time.time() - start_time

        metrics['elapsed_seconds'] = elapsed
        metrics['phase'] = phase
        metrics['model'] = "BAAI/bge-base-en-v1.5"
        metrics['embedding_dim'] = evaluator.embedding_dim

    # Print results
    evaluator.print_results(metrics, phase=phase)
    print(f"Evaluation time: {elapsed:.1f} seconds")

    # Save results
    if args.output:
        evaluator.save_results(metrics, args.output)
    else:
        # Default output path
        output_path = f"tests/benchmarks/fixtures/hotpotqa_phase{phase}_results.json"
        evaluator.save_results(metrics, output_path)


if __name__ == '__main__':
    main()
