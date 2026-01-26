#!/usr/bin/env python3
"""Detailed debug script to understand query() internals"""

import sys
from pathlib import Path
import tempfile
import os
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemefusion-python" / ".venv" / "Lib" / "site-packages"))

import mnemefusion

# Create a temporary database
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "debug.mfdb")

    config = {"embedding_dim": 3}
    engine = mnemefusion.Memory(db_path, config=config)

    # Add test memories with timestamps
    print("Adding test memories...")
    now = time.time()

    id1 = engine.add(
        content="The sky is blue",
        embedding=[0.1, 0.2, 0.3],
        metadata={"topic": "sky", "color": "blue"},
        timestamp=now - 100,  # 100 seconds ago
        source=None,
        namespace=None
    )

    id2 = engine.add(
        content="The grass is green",
        embedding=[0.2, 0.3, 0.4],
        metadata={"topic": "grass", "color": "green"},
        timestamp=now - 50,  # 50 seconds ago
        source=None,
        namespace=None
    )

    id3 = engine.add(
        content="The ocean is blue",
        embedding=[0.15, 0.25, 0.35],
        metadata={"topic": "ocean", "color": "blue"},
        timestamp=now,  # just now
        source=None,
        namespace=None
    )

    print(f"Added memories: {id1}, {id2}, {id3}")
    print(f"Memory count: {engine.count()}")

    # Test search first (should work)
    print("\n=== Testing search() ===")
    query_embedding = [0.15, 0.25, 0.35]
    search_results = engine.search(
        query_embedding=query_embedding,
        top_k=3,
        namespace=None,
        filters=None
    )
    print(f"Search returned {len(search_results)} results")
    for i, (memory, score) in enumerate(search_results):
        print(f"  {i+1}. content=\"{memory['content']}\" score={score:.4f}")

    # Test query
    print("\n=== Testing query() ===")
    intent, query_results = engine.query(
        query_text="What color is the ocean?",
        query_embedding=query_embedding,
        limit=3,
        namespace=None,
        filters=None
    )

    print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
    print(f"Query returned {len(query_results)} results")

    if len(query_results) == 0:
        print("\n!!! ERROR: query() returned 0 results while search() returned {} results !!!".format(len(search_results)))
        print("\nPossible causes:")
        print("1. Temporal search returning empty results")
        print("2. Entity search causing issues")
        print("3. Fusion engine filtering out all results")
        print("4. Bug in query planner")
    else:
        for i, (memory, scores) in enumerate(query_results):
            print(f"  {i+1}. content=\"{memory['content']}\"")
            print(f"     Semantic: {scores['semantic_score']:.4f}")
            print(f"     Temporal: {scores['temporal_score']:.4f}")
            print(f"     Causal: {scores['causal_score']:.4f}")
            print(f"     Entity: {scores['entity_score']:.4f}")
            print(f"     Fused: {scores['fused_score']:.4f}")
