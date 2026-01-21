"""
Basic usage example for MnemeFusion Python bindings

This example demonstrates:
- Creating a memory database
- Adding memories with embeddings
- Semantic search
- Intelligent query with intent classification
- Causal relationships
- Entity tracking
"""

import mnemefusion
import random


def random_embedding(dim=384):
    """Generate a random embedding for demonstration purposes"""
    return [random.random() for _ in range(dim)]


def main():
    print("=" * 60)
    print("MnemeFusion Python Example")
    print("=" * 60)
    print()

    # Create or open a memory database
    print("Opening memory database...")
    memory = mnemefusion.Memory("example_python.mfdb", config={
        "embedding_dim": 384,
        "entity_extraction_enabled": True,
    })

    print(f"Current memory count: {memory.count()}")
    print()

    # Add some memories
    print("Adding memories...")

    meeting_id = memory.add(
        "Team meeting scheduled for March 15th to discuss Project Alpha",
        random_embedding(),
        metadata={"type": "meeting", "project": "Alpha"}
    )
    print(f"  Added meeting: {meeting_id}")

    cancel_id = memory.add(
        "Meeting cancelled due to scheduling conflict with stakeholder review",
        random_embedding(),
        metadata={"type": "cancellation"}
    )
    print(f"  Added cancellation: {cancel_id}")

    notes_id = memory.add(
        "Alice provided detailed feedback on the API design for Project Alpha",
        random_embedding(),
        metadata={"type": "note", "author": "Alice"}
    )
    print(f"  Added notes: {notes_id}")

    print(f"\nTotal memories: {memory.count()}")
    print()

    # Add a causal relationship
    print("Adding causal relationship...")
    memory.add_causal_link(
        cancel_id,
        meeting_id,
        0.95,
        "Cancellation directly caused the meeting change"
    )
    print("  Linked: cancellation -> meeting")
    print()

    # Retrieve a memory
    print("Retrieving memory...")
    result = memory.get(meeting_id)
    if result:
        print(f"  Content: {result['content']}")
        print(f"  Created: {result['created_at']}")
        print(f"  Metadata: {result['metadata']}")
    print()

    # Semantic search
    print("Performing semantic search...")
    query_emb = random_embedding()
    search_results = memory.search(query_emb, top_k=3)

    print(f"  Found {len(search_results)} results:")
    for mem, score in search_results[:3]:
        print(f"    Score {score:.3f}: {mem['content'][:60]}...")
    print()

    # Intelligent query with intent classification
    print("Performing intelligent query...")
    print("  Query: 'Why was the meeting cancelled?'")

    intent, query_results = memory.query(
        "Why was the meeting cancelled?",
        random_embedding(),
        limit=5
    )

    print(f"  Intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
    print(f"  Results: {len(query_results)}")

    for mem, scores in query_results[:2]:
        print(f"    Fused score: {scores['fused_score']:.3f}")
        print(f"      Semantic: {scores['semantic_score']:.3f}")
        print(f"      Temporal: {scores['temporal_score']:.3f}")
        print(f"      Causal:   {scores['causal_score']:.3f}")
        print(f"      Entity:   {scores['entity_score']:.3f}")
        print(f"      Content: {mem['content'][:50]}...")
    print()

    # Get causal chains
    print("Exploring causal chains...")
    causes = memory.get_causes(meeting_id, max_hops=3)
    print(f"  Causes of meeting: {len(causes)} paths found")
    for i, path in enumerate(causes[:2]):
        print(f"    Path {i+1}: {len(path)} memories")
    print()

    # List entities
    print("Listing entities...")
    entities = memory.list_entities()
    print(f"  Found {len(entities)} entities:")
    for entity in entities[:5]:
        print(f"    - {entity['name']} (mentioned {entity['mention_count']} times)")
    print()

    # Delete a memory
    print("Deleting a memory...")
    deleted = memory.delete(notes_id)
    print(f"  Deleted: {deleted}")
    print(f"  New count: {memory.count()}")
    print()

    # Close the database
    print("Closing database...")
    memory.close()
    print("Done!")
    print()


def context_manager_example():
    """Example using context manager (with statement)"""
    print("=" * 60)
    print("Context Manager Example")
    print("=" * 60)
    print()

    with mnemefusion.Memory("example_context.mfdb") as memory:
        mem_id = memory.add(
            "This memory is added within a context manager",
            random_embedding()
        )
        print(f"Added memory: {mem_id}")
        print(f"Total: {memory.count()}")

    # Database is automatically closed when exiting the 'with' block
    print("Database automatically closed")
    print()


if __name__ == "__main__":
    main()
    context_manager_example()
