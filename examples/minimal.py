#!/usr/bin/env python3
"""
Minimal MnemeFusion example — no GPU, no GGUF model required.

Demonstrates multi-dimensional retrieval with RRF fusion using only
sentence-transformers for embeddings. Runs in ~20 seconds.

Prerequisites:
    pip install mnemefusion-cpu sentence-transformers
"""

import mnemefusion
from sentence_transformers import SentenceTransformer

# Load embedding model (~130MB on first run)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Open or create a database (768 = BGE-base embedding dimension)
mem = mnemefusion.Memory("./demo.mfdb", {"embedding_dim": 768})
mem.set_embedding_fn(lambda text: model.encode(text).tolist())

# Add some memories
memories = [
    ("Alice loves hiking in the Rocky Mountains every summer", {"speaker": "narrator"}),
    ("Bob started learning piano three months ago", {"speaker": "narrator"}),
    ("Alice and Bob met at a coffee shop in Denver in 2019", {"speaker": "narrator"}),
    ("Bob plays jazz piano at a local bar on weekends", {"speaker": "narrator"}),
    ("Alice completed a marathon in under 4 hours last October", {"speaker": "narrator"}),
    ("The coffee shop where they met closed down in 2022", {"speaker": "narrator"}),
    ("Alice is training for an ultramarathon next spring", {"speaker": "narrator"}),
    ("Bob's piano teacher recommended he try composing", {"speaker": "narrator"}),
    ("They go hiking together every month when the weather is good", {"speaker": "narrator"}),
    ("Bob composed his first original song last week", {"speaker": "narrator"}),
]

print(f"Adding {len(memories)} memories...")
for content, metadata in memories:
    mem.add(content, metadata=metadata)

# Query — multi-dimensional retrieval with RRF fusion
queries = [
    "What are Alice's hobbies?",
    "How did Alice and Bob meet?",
    "What has Bob been doing with music?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    intent, results, profiles = mem.query(query, limit=5)
    print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")

    for i, (memory_dict, scores_dict) in enumerate(results, 1):
        print(f"  {i}. [{scores_dict['fused_score']:.3f}] {memory_dict['content']}")

    if profiles:
        print(f"\n  Entity context:")
        for fact in profiles[:3]:
            print(f"    - {fact}")

# Clean up
mem.close()

# Remove demo database
import os
os.remove("./demo.mfdb")
print("\nDone! Demo database cleaned up.")
