#!/usr/bin/env python3
"""
Debug script for Sprint 16.4 regression investigation

This script tests individual queries to understand why 4D fusion is underperforming.
"""

import sys
from pathlib import Path

# Add parent directory to path for mnemefusion imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemefusion-python"))

try:
    from sentence_transformers import SentenceTransformer
    import mnemefusion
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)

import tempfile
import os


def test_query(engine, model, query_text, expected_intent=None):
    """Test a single query and print debug information"""
    print(f"\n{'='*80}")
    print(f"Query: {query_text}")
    print(f"Expected intent: {expected_intent}")
    print(f"{'='*80}")

    # Generate embedding
    query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()

    # Test semantic-only search (baseline)
    print("\n[SEMANTIC-ONLY SEARCH]")
    semantic_results = engine.search(
        query_embedding=query_embedding,
        top_k=5,
        namespace=None,
        filters=None
    )
    print(f"  Results: {len(semantic_results)}")
    if semantic_results:
        print(f"  Top score: {semantic_results[0][1]:.4f}")
        print(f"  Top content: {semantic_results[0][0]['content'][:100]}...")

    # Test 4D fusion query
    print("\n[4D FUSION QUERY]")
    try:
        intent, fusion_results = engine.query(
            query_text=query_text,
            query_embedding=query_embedding,
            limit=5,
            namespace=None,
            filters=None
        )

        print(f"  Detected intent: {intent}")
        print(f"  Results: {len(fusion_results)}")

        if fusion_results:
            print(f"\n  Top 3 results:")
            for i, (memory, scores) in enumerate(fusion_results[:3], 1):
                print(f"    {i}. Fused: {scores.get('fused_score', 0):.4f} "
                      f"| Sem: {scores.get('semantic_score', 0):.4f} "
                      f"| Temp: {scores.get('temporal_score', 0):.4f} "
                      f"| Ent: {scores.get('entity_score', 0):.4f} "
                      f"| Caus: {scores.get('causal_score', 0):.4f}")
                print(f"       {memory['content'][:80]}...")

        # Check if any dimension has non-zero scores
        if fusion_results:
            has_temporal = any(s.get('temporal_score', 0) > 0 for _, s in fusion_results)
            has_entity = any(s.get('entity_score', 0) > 0 for _, s in fusion_results)
            has_causal = any(s.get('causal_score', 0) > 0 for _, s in fusion_results)

            print(f"\n  Dimension activity:")
            print(f"    Semantic: ✅ (always active)")
            print(f"    Temporal: {'✅' if has_temporal else '❌ (no scores)'}")
            print(f"    Entity: {'✅' if has_entity else '❌ (no scores)'}")
            print(f"    Causal: {'✅' if has_causal else '❌ (no scores)'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*80)
    print("Sprint 16.4 Regression Debug Script")
    print("="*80)

    # Load embedding model
    print("\nLoading embedding model: BAAI/bge-base-en-v1.5")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "debug.mfdb")

        print(f"\nCreating database at: {db_path}")
        engine = mnemefusion.Memory(db_path, config={"embedding_dim": 768})

        # Add some test memories
        print("\nAdding test memories...")
        memories = [
            ("I went to the park yesterday with Alice to play tennis", "temporal + entity"),
            ("The meeting was cancelled because Bob was sick", "causal + entity"),
            ("Machine learning is a subset of artificial intelligence", "factual"),
            ("I had lunch with Carol at the Italian restaurant last week", "temporal + entity"),
            ("The bug was caused by a race condition in the database", "causal"),
            ("Python is a programming language created by Guido van Rossum", "factual + entity"),
        ]

        for content, description in memories:
            embedding = model.encode(content, convert_to_numpy=True).tolist()
            memory_id = engine.add(
                content=content,
                embedding=embedding,
                metadata={"description": description},
                timestamp=None
            )
            print(f"  Added: {description}")

        print(f"\nTotal memories: {len(memories)}")

        # Test queries
        test_cases = [
            ("What happened yesterday?", "Temporal"),
            ("Tell me about Alice", "Entity"),
            ("Why was the meeting cancelled?", "Causal"),
            ("What is machine learning?", "Factual"),
            ("When did I have lunch?", "Temporal"),
            ("What caused the bug?", "Causal"),
        ]

        for query, expected_intent in test_cases:
            test_query(engine, model, query, expected_intent)

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print("\nExpected behavior:")
        print("  - Temporal queries should have temporal_score > 0")
        print("  - Entity queries should have entity_score > 0")
        print("  - Causal queries should have causal_score > 0")
        print("  - 4D fusion should rank results differently than semantic-only")
        print("\nIf dimensions show '❌ (no scores)', that's the problem!")
        print("Check Sprint 16.1-16.3 implementations.")


if __name__ == "__main__":
    main()
