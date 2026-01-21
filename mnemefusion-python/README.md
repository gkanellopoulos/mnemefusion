# MnemeFusion Python Bindings

Python bindings for MnemeFusion - a unified memory engine for AI applications.

## Installation

### From Source (Development)

```bash
cd mnemefusion-python
pip install maturin
maturin develop
```

### From PyPI (when published)

```bash
pip install mnemefusion
```

## Quick Start

```python
import mnemefusion

# Open or create a database
memory = mnemefusion.Memory("brain.mfdb", config={
    "embedding_dim": 384,
    "entity_extraction_enabled": True,
})

# Add a memory
memory_id = memory.add(
    "Meeting scheduled for March 15th",
    embedding=[0.1] * 384,  # Your embedding vector
    metadata={"type": "event"}
)

# Semantic search
results = memory.search(query_embedding, top_k=10)
for mem, score in results:
    print(f"{score:.3f}: {mem['content']}")

# Intelligent query with intent classification
intent, results = memory.query(
    "Why was the meeting cancelled?",
    query_embedding,
    limit=10
)
print(f"Intent: {intent['intent']} ({intent['confidence']:.2f})")
for mem, scores in results:
    print(f"Fused score: {scores['fused_score']:.3f}")
    print(f"Content: {mem['content']}")

# Close the database
memory.close()
```

## Using Context Manager

```python
with mnemefusion.Memory("brain.mfdb") as memory:
    memory.add("Some content", embedding)
    # Database automatically closed when exiting
```

## API Reference

### Memory Class

#### `__init__(path, config=None)`

Create or open a memory database.

**Parameters:**
- `path` (str): Path to the .mfdb file
- `config` (dict, optional): Configuration options
  - `embedding_dim` (int): Dimension of embedding vectors (default: 384)
  - `entity_extraction_enabled` (bool): Enable automatic entity extraction (default: True)

**Returns:** Memory instance

#### `add(content, embedding, metadata=None, timestamp=None)`

Add a new memory to the database.

**Parameters:**
- `content` (str): Text content
- `embedding` (List[float]): Vector embedding
- `metadata` (Dict[str, str], optional): Key-value metadata
- `timestamp` (float, optional): Unix timestamp (seconds since epoch)

**Returns:** Memory ID as string

#### `get(memory_id)`

Retrieve a memory by ID.

**Parameters:**
- `memory_id` (str): Memory ID

**Returns:** Dictionary with memory data, or None if not found

**Memory dict structure:**
```python
{
    "id": str,
    "content": str,
    "embedding": List[float],
    "metadata": Dict[str, str],
    "created_at": float  # Unix timestamp
}
```

#### `delete(memory_id)`

Delete a memory by ID.

**Parameters:**
- `memory_id` (str): Memory ID

**Returns:** bool - True if deleted, False if not found

#### `search(query_embedding, top_k)`

Semantic similarity search.

**Parameters:**
- `query_embedding` (List[float]): Query vector
- `top_k` (int): Number of results to return

**Returns:** List of (memory_dict, similarity_score) tuples

#### `query(query_text, query_embedding, limit)`

Intelligent multi-dimensional query with intent classification.

**Parameters:**
- `query_text` (str): Natural language query
- `query_embedding` (List[float]): Query vector
- `limit` (int): Maximum number of results

**Returns:** Tuple of (intent_dict, results_list)
- `intent_dict`: `{"intent": str, "confidence": float}`
- `results_list`: List of (memory_dict, scores_dict) tuples

**Scores dict structure:**
```python
{
    "semantic_score": float,
    "temporal_score": float,
    "causal_score": float,
    "entity_score": float,
    "fused_score": float
}
```

**Intent types:**
- `"Temporal"`: Time-based queries ("yesterday", "recent")
- `"Causal"`: Cause-effect queries ("why", "because")
- `"Entity"`: Entity-focused queries ("about X", "mentioning Y")
- `"Factual"`: Generic semantic search

#### `count()`

Get the number of memories in the database.

**Returns:** int - Count of memories

#### `add_causal_link(cause_id, effect_id, confidence, evidence)`

Add a causal relationship between two memories.

**Parameters:**
- `cause_id` (str): Memory ID of the cause
- `effect_id` (str): Memory ID of the effect
- `confidence` (float): Confidence score (0.0 to 1.0)
- `evidence` (str): Text explaining the relationship

#### `get_causes(memory_id, max_hops)`

Get causes of a memory (backward traversal).

**Parameters:**
- `memory_id` (str): Memory ID
- `max_hops` (int): Maximum traversal depth

**Returns:** List of causal paths (each path is a list of memory ID strings)

#### `get_effects(memory_id, max_hops)`

Get effects of a memory (forward traversal).

**Parameters:**
- `memory_id` (str): Memory ID
- `max_hops` (int): Maximum traversal depth

**Returns:** List of causal paths (each path is a list of memory ID strings)

#### `list_entities()`

List all entities in the database.

**Returns:** List of entity dictionaries

**Entity dict structure:**
```python
{
    "id": str,
    "name": str,
    "mention_count": int,
    "metadata": Dict[str, str]
}
```

#### `close()`

Close the database and save all indexes.

## Examples

See [examples/basic_usage.py](examples/basic_usage.py) for a comprehensive example.

## Requirements

- Python >= 3.8
- Rust toolchain (for building from source)

## License

MIT
