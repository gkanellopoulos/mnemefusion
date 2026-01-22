# Getting Started with MnemeFusion

Welcome to MnemeFusion - a unified memory engine for AI applications. This guide will walk you through your first steps with MnemeFusion.

## What is MnemeFusion?

MnemeFusion is **"SQLite for AI memory"** - a single-file embedded database that provides four-dimensional memory indexing:

1. **Semantic** - Find similar memories by meaning (vector similarity)
2. **Temporal** - Find memories by time (recent, time ranges)
3. **Causal** - Trace cause-and-effect relationships
4. **Entity** - Find memories mentioning specific entities (people, places, things)

Unlike traditional solutions that require Qdrant + Neo4j + SQLite, MnemeFusion combines all four dimensions in one library, one file, with zero external dependencies.

---

## Installation

### Option 1: From Source (Development)

**Prerequisites:**
- Rust 1.75+ ([install here](https://rustup.rs/))
- Python 3.8+ (for Python bindings)

```bash
# Clone the repository
git clone https://github.com/yourusername/mnemefusion.git
cd mnemefusion

# Build Python bindings
cd mnemefusion-python
python -m venv .venv
.venv/Scripts/activate  # On Windows
# source .venv/bin/activate  # On Linux/macOS

pip install maturin
maturin develop

# Verify installation
python -c "import mnemefusion; print('MnemeFusion installed!')"
```

### Option 2: From PyPI (Coming Soon)

```bash
pip install mnemefusion  # Not yet published
```

---

## Quick Start: Your First Memory Database

### 1. Create a Memory Database

```python
import mnemefusion

# Create or open a database file
memory = mnemefusion.Memory("my_brain.mfdb")

# Check current state
print(f"Memories stored: {memory.count()}")
```

### 2. Add Memories with Embeddings

```python
# You need to provide embeddings (vectors that represent the semantic meaning)
# These typically come from an embedding model like OpenAI, Sentence Transformers, etc.

# Example: Generate random embeddings for demonstration
# In production, use a real embedding model!
import random

def get_embedding(text):
    """Placeholder - use a real embedding model in production"""
    return [random.random() for _ in range(384)]

# Add a memory
memory_id = memory.add(
    content="Team meeting scheduled for March 15th",
    embedding=get_embedding("Team meeting scheduled for March 15th"),
    metadata={"type": "event", "priority": "high"}
)

print(f"Added memory: {memory_id}")
```

### 3. Search by Semantic Similarity

```python
# Search for similar memories
query = "upcoming meetings"
query_embedding = get_embedding(query)

results = memory.search(query_embedding, top_k=5)

for mem, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {mem['content']}")
    print(f"Metadata: {mem['metadata']}")
    print()
```

### 4. Intelligent Query (Intent-Aware)

MnemeFusion automatically detects what kind of query you're making and weights the four dimensions accordingly.

```python
# Causal query - automatically emphasizes causal relationships
intent, results = memory.query(
    query_text="Why was the meeting cancelled?",
    query_embedding=get_embedding("Why was the meeting cancelled?"),
    limit=10
)

print(f"Detected intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")

for mem, scores in results:
    print(f"Fused score: {scores['fused_score']:.3f}")
    print(f"  Semantic: {scores['semantic_score']:.3f}")
    print(f"  Temporal: {scores['temporal_score']:.3f}")
    print(f"  Causal: {scores['causal_score']:.3f}")
    print(f"  Entity: {scores['entity_score']:.3f}")
    print(f"  Content: {mem['content']}")
    print()
```

### 5. Add Causal Relationships

```python
# Link cause and effect
meeting_id = memory.add(
    "Team meeting scheduled",
    get_embedding("Team meeting scheduled")
)

cancellation_id = memory.add(
    "Meeting cancelled due to conflict",
    get_embedding("Meeting cancelled due to conflict")
)

# Create causal link
memory.add_causal_link(
    cause_id=cancellation_id,
    effect_id=meeting_id,
    confidence=0.9,
    evidence="Email from manager explaining conflict"
)

# Traverse causal chains
causes = memory.get_causes(meeting_id, max_hops=3)
print(f"Causes found: {len(causes)} paths")
```

### 6. Work with Entities

```python
# Entities are automatically extracted from memory content
# (looks for capitalized words like names, places, organizations)

memory.add(
    "Alice provided feedback on Project Alpha API design",
    get_embedding("Alice provided feedback on Project Alpha API design")
)

# List all extracted entities
entities = memory.list_entities()
for entity in entities:
    print(f"{entity['name']}: mentioned {entity['mention_count']} times")
```

### 7. Clean Up

```python
# Always close the database when done
memory.close()

# Or use context manager (recommended)
with mnemefusion.Memory("my_brain.mfdb") as memory:
    # All operations here
    memory.add("Some content", get_embedding("Some content"))
    # Database automatically closes
```

---

## Real-World Integration: Using with Sentence Transformers

Here's a complete example using actual embeddings:

```python
from sentence_transformers import SentenceTransformer
import mnemefusion

# Load an embedding model (run once, cache it)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

def embed(text):
    return model.encode(text).tolist()

# Create memory database
with mnemefusion.Memory("brain.mfdb") as memory:
    # Add conversation history
    memory.add(
        "User prefers Italian food",
        embed("User prefers Italian food"),
        metadata={"type": "preference", "category": "food"}
    )

    memory.add(
        "User dislikes spicy dishes",
        embed("User dislikes spicy dishes"),
        metadata={"type": "preference", "category": "food"}
    )

    memory.add(
        "User asked about restaurant recommendations yesterday",
        embed("User asked about restaurant recommendations yesterday"),
        metadata={"type": "query", "category": "food"}
    )

    # Later: Search for relevant context
    user_query = "Where should I eat tonight?"

    intent, results = memory.query(
        query_text=user_query,
        query_embedding=embed(user_query),
        limit=5
    )

    print(f"Intent: {intent['intent']}")
    print(f"Relevant memories:")
    for mem, scores in results:
        print(f"  - {mem['content']} (score: {scores['fused_score']:.3f})")
```

---

## Key Concepts

### 1. Four Dimensions of Memory

MnemeFusion indexes memories across four independent dimensions:

| Dimension | What it finds | Example Query |
|-----------|---------------|---------------|
| **Semantic** | Similar meaning | "Tell me about the API" |
| **Temporal** | Time-based | "What happened yesterday?" |
| **Causal** | Cause-and-effect | "Why was the meeting cancelled?" |
| **Entity** | Mentions of people/things | "What did Alice say?" |

### 2. Intent Classification

When you use `query()`, MnemeFusion automatically detects intent:

- **Temporal** - "yesterday", "recent", "last week"
- **Causal** - "why", "because", "caused", "reason"
- **Entity** - "about X", "regarding Y", capitalized words
- **Factual** - Default semantic search

Intent determines how the four dimensions are weighted in results.

### 3. Adaptive Weights

Based on detected intent, MnemeFusion adjusts dimension weights:

```python
# Causal query: "Why did this happen?"
weights = {
    "semantic": 0.3,   # Meaning is somewhat relevant
    "temporal": 0.1,   # Time is less important
    "causal": 0.5,     # Causality is most important
    "entity": 0.1      # Entities are less important
}

# Temporal query: "What happened yesterday?"
weights = {
    "semantic": 0.3,
    "temporal": 0.5,   # Time is most important
    "causal": 0.1,
    "entity": 0.1
}
```

### 4. Single-File Database

Everything is stored in one `.mfdb` file:
- Memory content and metadata
- Vector embeddings (semantic index)
- Temporal index (B-tree)
- Causal graph (edges)
- Entity graph (bipartite)

No external services required. No network calls. Just one file.

---

## Configuration Options

```python
memory = mnemefusion.Memory(
    path="brain.mfdb",
    config={
        "embedding_dim": 384,                  # Match your embedding model
        "entity_extraction_enabled": True,     # Auto-extract entities
    }
)
```

---

## Common Patterns

### Pattern 1: Conversational Memory

```python
# Store each turn of conversation
with mnemefusion.Memory("conversation.mfdb") as memory:
    for turn in conversation_history:
        memory.add(
            content=turn.text,
            embedding=embed(turn.text),
            metadata={
                "speaker": turn.speaker,
                "turn_number": turn.number,
                "session_id": session_id
            },
            timestamp=turn.timestamp  # Optional: custom timestamp
        )

    # Later: Retrieve relevant context
    intent, context = memory.query(
        query_text=current_user_message,
        query_embedding=embed(current_user_message),
        limit=5
    )
```

### Pattern 2: Document Memory

```python
# Store document chunks with source tracking
with mnemefusion.Memory("documents.mfdb") as memory:
    for doc in documents:
        for chunk in doc.chunks:
            memory.add(
                content=chunk.text,
                embedding=embed(chunk.text),
                metadata={
                    "document_id": doc.id,
                    "chunk_id": chunk.id,
                    "page": chunk.page,
                    "source": doc.filename
                }
            )
```

### Pattern 3: Causal Reasoning

```python
# Build a knowledge graph with cause-effect relationships
with mnemefusion.Memory("knowledge.mfdb") as memory:
    # Add events
    event1_id = memory.add("Economy grew 3%", embed("Economy grew 3%"))
    event2_id = memory.add("Interest rates lowered", embed("Interest rates lowered"))
    event3_id = memory.add("Stock market rose", embed("Stock market rose"))

    # Link causes
    memory.add_causal_link(event2_id, event1_id, 0.7, "Economic policy")
    memory.add_causal_link(event1_id, event3_id, 0.8, "Market confidence")

    # Traverse chains
    effects = memory.get_effects(event2_id, max_hops=3)
    print(f"Downstream effects: {effects}")
```

---

## Next Steps

- **Explore Examples**: Check `examples/` directory for more use cases
- **Read API Docs**: See `mnemefusion-python/README.md` for complete API reference
- **Join Community**: [Link to Discord/Discussions when available]
- **Report Issues**: [GitHub Issues](https://github.com/yourusername/mnemefusion/issues)

---

## Troubleshooting

### "Module not found: mnemefusion"

Make sure you installed with `maturin develop` in the Python environment you're using.

```bash
cd mnemefusion-python
source .venv/bin/activate  # Activate your virtualenv
maturin develop
python -c "import mnemefusion"  # Should work now
```

### "Wrong embedding dimension"

Your embedding dimension must match the config:

```python
memory = mnemefusion.Memory("brain.mfdb", config={"embedding_dim": 384})
memory.add("text", [0.1] * 384)  # Must be 384 dimensions
```

### "Database file locked"

Only one process can write to a database at a time. Make sure you close connections:

```python
memory.close()  # Always close when done
# Or use context manager to auto-close
```

---

## Performance Tips

1. **Batch Operations**: Group additions when possible (coming in Phase 2)
2. **Appropriate Dimensions**: Use smaller embeddings (384) for speed, larger (1536) for accuracy
3. **Entity Extraction**: Disable if not needed: `config={"entity_extraction_enabled": False}`
4. **Limit Results**: Use `top_k` or `limit` to avoid scanning entire database

---

## What's Next?

MnemeFusion is actively developed. Upcoming features (Phase 2):

- **Provenance Tracking**: Know where each memory came from
- **Batch Operations**: 10x faster bulk imports
- **Deduplication**: Automatic duplicate prevention
- **Namespaces**: Multi-user/multi-context isolation
- **Metadata Filtering**: Filter by metadata fields

See `IMPLEMENTATION_PLAN.md` for the full roadmap.

---

**Built with ❤️ by the MnemeFusion team**

Questions? Open an issue on [GitHub](https://github.com/yourusername/mnemefusion)
