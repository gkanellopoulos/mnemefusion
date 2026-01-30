#!/usr/bin/env python3
"""
Validation Experiment with SLM-Based Intent Classification

This script extends the validation experiment to use SLM (Small Language Model)
for intent classification instead of pattern-based classification.

Comparison:
- baseline (validation_eval.py): Uses pattern-based classification
- this script: Uses SLM classification via local Gemma-2-2B model

Strategy:
- Load LoCoMo benchmark dataset
- Build memory database with SLM classification enabled
- Run queries and measure recall, MRR, precision
- Compare with pattern-based baseline results
"""

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List

# Set MODEL_PATH if not already set
if "MODEL_PATH" not in os.environ:
    model_path = Path("opt/models/gemma-2-2b-it").absolute()
    if model_path.exists():
        os.environ["MODEL_PATH"] = str(model_path)
        print(f"[INFO] Set MODEL_PATH to: {model_path}")
    else:
        print(f"[WARNING] MODEL_PATH not set and {model_path} doesn't exist")
        print("[WARNING] SLM will attempt to download from HuggingFace")

# Reuse the existing evaluator
sys.path.insert(0, str(Path(__file__).parent))
from locomo_eval import LoCoMoEvaluator

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed. Run:")
    print("  cd mnemefusion-python && maturin develop --features slm --release")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='LoCoMo Validation with SLM Classification'
    )
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of conversations (1-10)')
    parser.add_argument('--dataset', type=str,
                        default='fixtures/locomo10.json',
                        help='Path to LoCoMo dataset')
    parser.add_argument('--output', type=str,
                        required=True,
                        help='Output JSON file for results')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU for embeddings')
    parser.add_argument('--model-path', type=str,
                        help='Path to local SLM model (overrides MODEL_PATH env var)')

    args = parser.parse_args()

    # Override MODEL_PATH if provided
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
        print(f"[INFO] Using model path: {args.model_path}")

    # Initialize evaluator (no overrides needed - using real SLM)
    evaluator = LoCoMoEvaluator(use_gpu=not args.no_gpu)

    # Load dataset
    conversations = evaluator.load_dataset(args.dataset, args.samples)

    # Use evaluator's parsing methods
    documents = evaluator.prepare_documents(conversations)
    qa_pairs = evaluator.extract_qa_pairs(conversations)

    print(f"\n[INFO] Loaded {len(conversations)} conversations")
    print(f"[INFO] Total turns: {len(documents)}")
    print(f"[INFO] Total QA pairs: {len(qa_pairs)}")

    # Generate embeddings
    print("\n[INFO] Generating embeddings...")
    # Documents are tuples: (doc_id, content, metadata)
    doc_texts = [doc[1] for doc in documents]  # Extract content
    doc_embeddings = evaluator.generate_embeddings(doc_texts)

    print(f"[INFO] Generated {len(doc_embeddings)} embeddings")

    # Build database WITH SLM
    print("\n[INFO] Building memory database WITH SLM classification...")
    print("[INFO] This enables local Gemma-2-2B model for intent classification")

    # Create temp directory for database (don't pre-create the file)
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "validation.mfdb")

    try:
        # Create engine with SLM enabled
        engine = mnemefusion.Memory(
            db_path,
            config={
                "embedding_dim": evaluator.embedding_dim,
                "use_slm": True  # Enable SLM classification!
            }
        )

        print("[INFO] SLM classification enabled")
        print(f"[INFO] Model path: {os.environ.get('MODEL_PATH', 'AUTO-DOWNLOAD')}")

        # Reserve capacity for all documents
        engine.reserve_capacity(len(documents) + 1000)  # Add buffer
        print(f"[INFO] Reserved capacity for {len(documents)} documents")

        # Add documents
        print("[INFO] Adding documents to database...")
        for i, ((doc_id, content, metadata), embedding) in enumerate(zip(documents, doc_embeddings)):
            # Convert metadata values to strings
            metadata_str = {k: str(v) for k, v in metadata.items()}

            engine.add(
                content=content,
                embedding=embedding.tolist(),
                metadata=metadata_str,
                timestamp=None,
                source=None,
                namespace=None
            )
            if (i + 1) % 100 == 0 or (i + 1) == len(documents):
                print(f"  Added {i + 1}/{len(documents)} turns")

        print(f"[OK] Database built with {len(documents)} conversation turns")

        # Run evaluation
        print("\n[INFO] Running evaluation with SLM classification...")
        start_time = time.time()
        results = evaluator.evaluate(qa_pairs, engine, top_k=10)
        elapsed = time.time() - start_time

        results['elapsed_seconds'] = elapsed
        results['phase'] = "2"
        results['model'] = "BAAI/bge-base-en-v1.5"
        results['embedding_dim'] = evaluator.embedding_dim
        results['num_conversations'] = len(conversations)
        results['classification_method'] = "slm"
        results['slm_model'] = "google/gemma-2-2b-it"
        results['model_path'] = os.environ.get('MODEL_PATH', 'auto-download')

        # Print results
        print("\n" + "="*60)
        print("Validation with SLM Classification - Phase 2")
        print("="*60)
        print(f"Classification:       SLM (google/gemma-2-2b-it)")
        print(f"Queries evaluated:    {results['num_queries']}")
        print(f"Top-K retrieved:      {results['top_k']}")
        print()
        print("Overall Metrics:")
        print(f"  Recall@10:         {results['recall_at_k']:.3f} ({results['recall_at_k']:.1%})")
        print(f"  MRR:                {results['mrr']:.3f}")
        print(f"  Precision@10:      {results['precision_at_k']:.3f} ({results['precision_at_k']:.1%})")
        print()

        target_recall = 0.70
        if results['recall_at_k'] >= target_recall:
            print(f"[TARGET MET] Recall@10 ({results['recall_at_k']:.3f}) >= target ({target_recall})")
        else:
            print(f"[BELOW TARGET] Recall@10 ({results['recall_at_k']:.3f}) < target ({target_recall})")

        print("="*60)
        print(f"Evaluation time: {elapsed:.1f} seconds")

        # Close engine
        engine.close()

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {args.output}")

    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
