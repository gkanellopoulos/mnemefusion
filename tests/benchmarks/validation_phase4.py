#!/usr/bin/env python3
"""
Phase 4 Validation: SLM Metadata at Ingestion Time

This script validates the new architecture:
- Phase 1: SLM metadata extraction at ingestion (3-5s per memory)
- Phase 2: Enhanced retrieval using SLM metadata
- Phase 3: No SLM at query time (fast queries)

Metrics to measure:
- Ingestion time (per memory and total)
- Query time (should be <100ms)
- Recall@10 (target: 70%+)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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

try:
    import mnemefusion
except ImportError:
    print("ERROR: mnemefusion not installed. Run: pip install -e mnemefusion-python/")
    sys.exit(1)


class Phase4Evaluator:
    """Evaluator for Phase 4 validation"""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", use_gpu: bool = True):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("[WARNING] GPU not available, using CPU")
            except ImportError:
                print("[WARNING] PyTorch not found, using CPU")

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """Generate embeddings for texts"""
        if instruction:
            texts = [instruction + t for t in texts]
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def load_dataset(self, dataset_path: str, num_conversations: int = None) -> List[Dict]:
        """Load LoCoMo dataset"""
        print(f"\nLoading dataset from {dataset_path}...")

        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found at {dataset_path}")
            sys.exit(1)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = data if isinstance(data, list) else [data]
        if num_conversations:
            conversations = conversations[:num_conversations]

        print(f"[OK] Loaded {len(conversations)} conversations")
        return conversations

    def prepare_documents(self, conversations: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Prepare conversation turns as documents"""
        print("\nPreparing documents...")
        documents = []

        for conv_idx, conversation in enumerate(conversations):
            sample_id = conversation.get('sample_id', f'conv_{conv_idx}')
            conv_data = conversation.get('conversation', {})

            session_idx = 1
            while f'session_{session_idx}' in conv_data:
                session_key = f'session_{session_idx}'
                turns = conv_data.get(session_key, [])

                for turn_idx, turn in enumerate(turns):
                    text = turn.get('text', '')
                    dialog_id = turn.get('dia_id', f'{sample_id}_s{session_idx}_t{turn_idx}')

                    if not text:
                        continue

                    metadata = {
                        'conversation_id': sample_id,
                        'session_idx': str(session_idx),
                        'turn_idx': str(turn_idx),
                        'speaker': turn.get('speaker', 'unknown'),
                        'dialog_id': dialog_id
                    }

                    documents.append((dialog_id, text, metadata))

                session_idx += 1

        print(f"[OK] Prepared {len(documents)} documents")
        return documents

    def extract_qa_pairs(self, conversations: List[Dict]) -> List[Dict]:
        """Extract QA pairs from conversations"""
        print("\nExtracting QA pairs...")
        qa_pairs = []

        for conversation in conversations:
            sample_id = conversation.get('sample_id', 'unknown')
            # LoCoMo uses 'qa' key, not 'qa_pairs'
            qa_list = conversation.get('qa', conversation.get('qa_pairs', []))

            for qa in qa_list:
                # LoCoMo uses 'evidence' key with dialog IDs like "D1:3"
                evidence = qa.get('evidence', qa.get('evidence_dial_ids', qa.get('evidence_dialog_ids', [])))
                qa_pairs.append({
                    'question': qa.get('question', ''),
                    'answer': qa.get('answer', ''),
                    'category': qa.get('category', 'unknown'),
                    'evidence_dialog_ids': evidence,
                    'conversation_id': sample_id
                })

        print(f"[OK] Extracted {len(qa_pairs)} QA pairs")
        return qa_pairs

    def run_validation(
        self,
        db_path: str,
        documents: List[Tuple[str, str, Dict]],
        qa_pairs: List[Dict],
        use_slm_extraction: bool = True,
        top_k: int = 10
    ) -> Dict:
        """Run full validation with metrics collection"""

        results = {
            'phase': '4',
            'architecture': 'slm_at_ingestion',
            'slm_extraction_enabled': use_slm_extraction,
            'top_k': top_k,
            'num_documents': len(documents),
            'num_queries': len(qa_pairs),
        }

        # Configure engine
        config = {
            "embedding_dim": self.embedding_dim,
            # SLM query classification is disabled by default (Phase 3)
            "slm_query_classification_enabled": False,
        }

        # Enable SLM metadata extraction at ingestion if requested
        if use_slm_extraction:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "opt" / "models" / "qwen3-0.6b.gguf"
            script_path = project_root / "scripts" / "slm_extract_server.py"

            if model_path.exists():
                print(f"[SLM] Using model: {model_path}")
                os.environ["SLM_EXTRACT_SCRIPT_PATH"] = str(script_path)
                config["use_slm"] = True
                config["slm_model_path"] = str(model_path)
                config["slm_metadata_extraction_enabled"] = True
            else:
                print(f"[WARNING] SLM model not found at {model_path}")
                print("[WARNING] Running without SLM metadata extraction")

        print(f"\nCreating MnemeFusion database at {db_path}...")
        engine = mnemefusion.Memory(str(db_path), config=config)
        engine.reserve_capacity(len(documents))

        # Generate embeddings for all documents
        print("Generating document embeddings...")
        embed_start = time.time()
        doc_texts = [doc[1] for doc in documents]
        doc_embeddings = self.generate_embeddings(doc_texts)
        embed_time = time.time() - embed_start
        results['embedding_time_seconds'] = embed_time
        print(f"[OK] Embeddings generated in {embed_time:.1f}s")

        # Ingest documents and measure time
        print("\nIngesting documents (with SLM metadata extraction if enabled)...")
        ingest_start = time.time()
        ingest_times = []

        for i, (dialog_id, content, metadata) in enumerate(documents):
            metadata_str = {k: str(v) for k, v in metadata.items()}

            doc_start = time.time()
            engine.add(
                content=content,
                embedding=doc_embeddings[i].tolist(),
                metadata=metadata_str,
                timestamp=None,
                source=None,
                namespace=None
            )
            doc_time = time.time() - doc_start
            ingest_times.append(doc_time)

            if (i + 1) % 100 == 0 or (i + 1) == len(documents):
                avg_time = np.mean(ingest_times[-100:]) if len(ingest_times) >= 100 else np.mean(ingest_times)
                print(f"  Ingested {i + 1}/{len(documents)} documents (avg: {avg_time*1000:.1f}ms/doc)")

        total_ingest_time = time.time() - ingest_start
        results['ingestion_time_seconds'] = total_ingest_time
        results['ingestion_time_per_doc_ms'] = (total_ingest_time / len(documents)) * 1000
        print(f"[OK] Ingestion complete in {total_ingest_time:.1f}s ({results['ingestion_time_per_doc_ms']:.1f}ms/doc)")

        # Run queries and measure recall
        print("\nEvaluating queries...")
        query_instruction = "Represent this sentence for searching relevant passages: "

        recall_scores = []
        mrr_scores = []
        precision_scores = []
        query_times = []
        category_metrics = {}

        for i, qa in enumerate(qa_pairs):
            query = qa['question']
            evidence_ids = set(qa['evidence_dialog_ids'])
            category = str(qa['category'])

            # Generate query embedding
            query_embedding = self.generate_embeddings([query], instruction=query_instruction)[0]

            # Run query and measure time
            query_start = time.time()
            detected_intent, query_results = engine.query(
                query_text=query,
                query_embedding=query_embedding.tolist(),
                limit=top_k,
                namespace=None,
                filters=None
            )
            query_time = (time.time() - query_start) * 1000  # ms
            query_times.append(query_time)

            # Extract retrieved dialog IDs
            retrieved_ids = []
            for memory, scores in query_results:
                dialog_id = memory['metadata'].get('dialog_id', '')
                retrieved_ids.append(dialog_id)

            # Calculate metrics
            relevant_retrieved = [id for id in retrieved_ids if id in evidence_ids]
            recall = len(relevant_retrieved) / len(evidence_ids) if evidence_ids else 0
            recall_scores.append(recall)

            # MRR
            first_relevant_rank = None
            for rank, id in enumerate(retrieved_ids, 1):
                if id in evidence_ids:
                    first_relevant_rank = rank
                    break
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            mrr_scores.append(mrr)

            # Precision
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            precision_scores.append(precision)

            # Track by category
            if category not in category_metrics:
                category_metrics[category] = {'recall': [], 'mrr': [], 'precision': [], 'count': 0}
            category_metrics[category]['recall'].append(recall)
            category_metrics[category]['mrr'].append(mrr)
            category_metrics[category]['precision'].append(precision)
            category_metrics[category]['count'] += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(qa_pairs):
                print(f"  Evaluated {i + 1}/{len(qa_pairs)} queries")

        # Aggregate results
        if recall_scores:
            results['recall_at_k'] = float(np.mean(recall_scores))
            results['mrr'] = float(np.mean(mrr_scores))
            results['precision_at_k'] = float(np.mean(precision_scores))
            results['query_time_avg_ms'] = float(np.mean(query_times))
            results['query_time_p50_ms'] = float(np.percentile(query_times, 50))
            results['query_time_p95_ms'] = float(np.percentile(query_times, 95))
            results['query_time_p99_ms'] = float(np.percentile(query_times, 99))
        else:
            results['recall_at_k'] = 0.0
            results['mrr'] = 0.0
            results['precision_at_k'] = 0.0
            results['query_time_avg_ms'] = 0.0
            results['query_time_p50_ms'] = 0.0
            results['query_time_p95_ms'] = 0.0
            results['query_time_p99_ms'] = 0.0

        # Category breakdown
        results['category_metrics'] = {}
        for cat, metrics in category_metrics.items():
            results['category_metrics'][cat] = {
                'recall_at_k': float(np.mean(metrics['recall'])),
                'mrr': float(np.mean(metrics['mrr'])),
                'precision_at_k': float(np.mean(metrics['precision'])),
                'count': metrics['count']
            }

        return results


def print_results(results: Dict):
    """Pretty print validation results"""
    print("\n" + "=" * 70)
    print("Phase 4 Validation Results: SLM at Ingestion Time")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Documents:              {results['num_documents']}")
    print(f"  Queries:                {results['num_queries']}")
    print(f"  SLM extraction:         {results['slm_extraction_enabled']}")
    print(f"  Top-K:                  {results['top_k']}")

    print(f"\nIngestion Performance:")
    print(f"  Total time:             {results['ingestion_time_seconds']:.1f}s")
    print(f"  Per document:           {results['ingestion_time_per_doc_ms']:.1f}ms")

    print(f"\nQuery Performance:")
    print(f"  Average:                {results['query_time_avg_ms']:.1f}ms")
    print(f"  P50:                    {results['query_time_p50_ms']:.1f}ms")
    print(f"  P95:                    {results['query_time_p95_ms']:.1f}ms")
    print(f"  P99:                    {results['query_time_p99_ms']:.1f}ms")

    print(f"\nRetrieval Metrics:")
    print(f"  Recall@{results['top_k']}:              {results['recall_at_k']:.3f} ({results['recall_at_k']*100:.1f}%)")
    print(f"  MRR:                    {results['mrr']:.3f}")
    print(f"  Precision@{results['top_k']}:          {results['precision_at_k']:.3f}")

    print(f"\nCategory Breakdown:")
    for cat, metrics in sorted(results['category_metrics'].items()):
        print(f"  Category {cat}: Recall={metrics['recall_at_k']:.1%}, MRR={metrics['mrr']:.3f}, Count={metrics['count']}")

    print("\n" + "=" * 70)

    # Check targets
    target_recall = 0.70
    target_query_time = 100  # ms

    if results['recall_at_k'] >= target_recall:
        print(f"[TARGET MET] Recall@{results['top_k']} ({results['recall_at_k']:.1%}) >= {target_recall:.0%}")
    else:
        print(f"[BELOW TARGET] Recall@{results['top_k']} ({results['recall_at_k']:.1%}) < {target_recall:.0%}")

    if results['query_time_avg_ms'] <= target_query_time:
        print(f"[TARGET MET] Query time ({results['query_time_avg_ms']:.1f}ms) <= {target_query_time}ms")
    else:
        print(f"[BELOW TARGET] Query time ({results['query_time_avg_ms']:.1f}ms) > {target_query_time}ms")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Validation: SLM at Ingestion')
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of conversations to process (1-10)')
    parser.add_argument('--dataset', type=str,
                        default='tests/benchmarks/fixtures/locomo10.json',
                        help='Path to LoCoMo dataset')
    parser.add_argument('--output', type=str,
                        default='tests/benchmarks/fixtures/phase4_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--no-slm', action='store_true',
                        help='Disable SLM metadata extraction (baseline comparison)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU for embeddings')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to retrieve')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Phase4Evaluator(use_gpu=not args.no_gpu)

    # Load dataset
    conversations = evaluator.load_dataset(args.dataset, args.samples)
    documents = evaluator.prepare_documents(conversations)
    qa_pairs = evaluator.extract_qa_pairs(conversations)

    # Run validation
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "phase4_eval.mfdb"

        results = evaluator.run_validation(
            db_path=str(db_path),
            documents=documents,
            qa_pairs=qa_pairs,
            use_slm_extraction=not args.no_slm,
            top_k=args.top_k
        )

        results['model'] = 'BAAI/bge-base-en-v1.5'
        results['embedding_dim'] = evaluator.embedding_dim
        results['num_conversations'] = len(conversations)

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to: {output_path}")


if __name__ == '__main__':
    main()
