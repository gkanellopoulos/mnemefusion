#!/usr/bin/env python3
"""Debug script to understand query() return format"""

import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mnemefusion-python" / ".venv" / "Lib" / "site-packages"))

import mnemefusion
import numpy as np

# Create a temporary database
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "debug.mfdb")

    config = {"embedding_dim": 3}
    engine = mnemefusion.Memory(db_path, config=config)

    # Add some test memories
    print("Adding test memories...")
    id1 = engine.add(
        content="The sky is blue",
        embedding=[0.1, 0.2, 0.3],
        metadata={"topic": "sky", "color": "blue"},
        timestamp=None,
        source=None,
        namespace=None
    )

    id2 = engine.add(
        content="The grass is green",
        embedding=[0.2, 0.3, 0.4],
        metadata={"topic": "grass", "color": "green"},
        timestamp=None,
        source=None,
        namespace=None
    )

    print(f"Added memories: {id1}, {id2}")

    # Test query
    print("\nTesting query()...")
    query_embedding = [0.15, 0.25, 0.35]
    intent, results = engine.query(
        query_text="What color is the sky?",
        query_embedding=query_embedding,
        limit=2,
        namespace=None,
        filters=None
    )

    print(f"\nIntent: {intent}")
    print(f"Intent type: {type(intent)}")
    print(f"Intent keys: {intent.keys() if isinstance(intent, dict) else 'Not a dict'}")

    print(f"\nResults: {type(results)}")
    print(f"Results length: {len(results)}")

    for i, result in enumerate(results):
        print(f"\n--- Result {i} ---")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")

        if len(result) == 2:
            memory, scores = result
            print(f"\nMemory type: {type(memory)}")
            print(f"Memory keys: {memory.keys() if isinstance(memory, dict) else 'Not a dict'}")
            if isinstance(memory, dict):
                print(f"Memory id: {memory.get('id', 'NO ID')}")
                print(f"Memory content: {memory.get('content', 'NO CONTENT')}")
                print(f"Memory metadata: {memory.get('metadata', 'NO METADATA')}")

            print(f"\nScores type: {type(scores)}")
            print(f"Scores: {scores}")

    # Test search for comparison
    print("\n\n=== Testing search() for comparison ===")
    search_results = engine.search(
        query_embedding=query_embedding,
        top_k=2,
        namespace=None,
        filters=None
    )

    print(f"Search results type: {type(search_results)}")
    print(f"Search results length: {len(search_results)}")

    for i, result in enumerate(search_results):
        print(f"\n--- Search Result {i} ---")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")

        if len(result) == 2:
            memory, score = result
            print(f"\nMemory type: {type(memory)}")
            print(f"Memory keys: {memory.keys() if isinstance(memory, dict) else 'Not a dict'}")
            if isinstance(memory, dict):
                print(f"Memory id: {memory.get('id', 'NO ID')}")
                print(f"Memory content: {memory.get('content', 'NO CONTENT')}")
                print(f"Memory metadata: {memory.get('metadata', 'NO METADATA')}")

            print(f"\nScore type: {type(score)}")
            print(f"Score: {score}")
