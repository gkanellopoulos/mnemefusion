# MnemeFusion: Feature Roadmap Extensions

## Features Informed by Competitive Analysis

**Document Version:** 0.1  
**Created:** January 2025  
**Status:** Planning Reference

---

## Overview

This document describes features that would enhance MnemeFusion's utility as a foundation for AI memory systems. These features were identified through competitive analysis (particularly Cognee) and are evaluated against MnemeFusion's core philosophy:

- ✅ Remains an embedded engine (no external services)
- ✅ No LLM dependency for core operations
- ✅ Single-file deployment model
- ✅ Simple, focused API

All features in this document pass these criteria.

---

## Priority Classification

| Priority | Meaning | Timeline |
|----------|---------|----------|
| **P0: Good to Have Now** | Essential for v1.0, blocks adoption without them | Phase 1 |
| **P1: Good to Have Soon** | Important for real-world usage, Phase 2 target | Phase 2 |
| **P2: Good to Have Later** | Nice to have, can wait for v1.x releases | Post-launch |

---

## P0: Good to Have Now

These features are essential for MnemeFusion to be useful as a foundation layer. Without them, developers will immediately hit walls in real-world usage.

---

### Feature: Deduplication / Upsert

**Priority:** P0 - Essential  
**Effort:** Medium (2-3 days)  
**Affects:** Storage layer, API

#### Problem Statement

Without deduplication, the same information stored twice creates pollution:

```python
# User says "My budget is $5,000" in turn 3
memory.add("Budget is $5,000", embedding)

# User repeats it in turn 7
memory.add("Budget is $5,000", embedding)  # Duplicate stored!

# Later retrieval returns both, confusing the system
```

Real-world scenarios where this matters:
- Retry logic after network failures
- User repeats information across sessions
- System restarts mid-conversation
- Batch imports with potential overlaps

#### Proposed Solution

**Option A: Content-Hash Deduplication (Automatic)**

```python
result = memory.add(content, embedding, dedup=True)

# Result indicates what happened
class AddResult:
    id: str                    # The memory ID (new or existing)
    created: bool              # True if new, False if duplicate
    existing_id: Optional[str] # If duplicate, the existing ID
```

Implementation:
- Hash content on add
- Check hash index before inserting
- Return existing ID if duplicate found
- Optional: update timestamp on duplicate (touch)

**Option B: Key-Based Upsert (Explicit)**

```python
# Explicit logical key
memory.upsert(
    key="user_budget",           # Developer-defined identifier
    content="Budget is $7,000",
    embedding=embedding
)

class UpsertResult:
    id: str
    created: bool               # True if new
    updated: bool               # True if replaced existing
    previous_content: Optional[str]  # What was replaced
```

Implementation:
- Secondary index on logical keys
- Lookup by key, update if exists, insert if not
- Atomic operation

**Recommendation:** Implement both.
- Content-hash dedup as default safety net
- Explicit upsert for intentional "this supersedes that" patterns

#### API Design

```python
class Memory:
    def add(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        dedup: bool = True,           # NEW: Enable content-hash dedup
    ) -> AddResult: ...
    
    def upsert(                        # NEW METHOD
        self,
        key: str,                      # Logical identifier
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> UpsertResult: ...
```

#### Storage Requirements

```
New index needed:

content_hash_index:
  hash(content) -> memory_id

logical_key_index:
  key -> memory_id
```

#### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Same content, different embedding | Treat as duplicate (content is source of truth) |
| Same key, different content | Upsert replaces content and embedding |
| Dedup=False | Allow duplicates (escape hatch) |
| Hash collision | Use full content comparison as fallback |

---

### Feature: Provenance / Source Tracking

**Priority:** P0 - Essential  
**Effort:** Low (1-2 days)  
**Affects:** Data model, API

#### Problem Statement

When AI retrieves a memory, users/developers need to know:
- Where did this information come from?
- Can I trust it?
- How do I verify it?
- When was the source created?

Without provenance:
```python
result = memory.search("What's the budget?")
# Returns: "Budget is $5,000"
# But... where did this come from? User said it? Document? Inferred?
```

#### Proposed Solution

Add a structured `source` field to memories:

```python
memory.add(
    content="Budget is $5,000",
    embedding=embedding,
    source={
        "type": "conversation",        # conversation | document | inference | manual
        "id": "conv_abc123",           # External reference
        "location": "turn_5",          # Where in the source
        "timestamp": "2025-01-15T10:00:00Z",
        "original_text": "User: My budget is around five thousand dollars",
        "confidence": 0.95,            # Optional: extraction confidence
        "extractor": "user_statement", # Optional: what extracted this
    }
)
```

#### Source Schema

```python
@dataclass
class Source:
    type: str                          # Required: conversation, document, url, manual, inference
    id: Optional[str] = None           # External identifier
    location: Optional[str] = None     # Position within source
    timestamp: Optional[str] = None    # When source was created
    original_text: Optional[str] = None  # Raw source text
    confidence: Optional[float] = None # 0.0-1.0 extraction confidence
    extractor: Optional[str] = None    # What process created this
    metadata: Optional[Dict] = None    # Extensible additional fields
```

#### Retrieval with Provenance

```python
results = memory.search(query, embedding)

for result in results:
    print(f"Content: {result.content}")
    print(f"Source: {result.source.type} / {result.source.id}")
    print(f"Original: {result.source.original_text}")
    print(f"Confidence: {result.source.confidence}")
```

#### Use Cases

| Use Case | How Provenance Helps |
|----------|---------------------|
| Debugging retrieval | "Why did it return this?" → See source |
| User trust | "The AI said X" → Show where X came from |
| Compliance/audit | Full traceability to source |
| Confidence filtering | Only use high-confidence memories |
| Source-type filtering | "Only from documents, not conversations" |

#### Storage Requirements

```
Extend memory record:

memory_record:
  id: MemoryId
  content: String
  embedding: Vec<f32>
  metadata: Map<String, String>
  source: Option<Source>           # NEW
  created_at: Timestamp
  updated_at: Timestamp
```

No new indexes required (source is stored, not indexed by default).

Optional future enhancement: Index by source.type for filtered retrieval.

---

### Feature: Batch Operations

**Priority:** P0 - Essential  
**Effort:** Low (1-2 days)  
**Affects:** API, performance

#### Problem Statement

Single-item operations are inefficient for bulk scenarios:

```python
# Terrible performance for startup/migration
for item in ten_thousand_items:
    memory.add(item.content, item.embedding)  # 10,000 separate transactions
```

Real-world scenarios:
- Initial data import
- Migration from another system
- Restoring from backup
- Batch processing pipelines

#### Proposed Solution

```python
memories = [
    MemoryInput(content="...", embedding=[...], metadata={...}, source={...}),
    MemoryInput(content="...", embedding=[...], metadata={...}, source={...}),
    # ... thousands more
]

result = memory.add_batch(memories)

class BatchResult:
    ids: List[str]               # Created memory IDs
    created_count: int           # How many new
    duplicate_count: int         # How many skipped (if dedup enabled)
    errors: List[BatchError]     # Any failures
```

#### Implementation Strategy

1. **Single transaction** - All inserts in one atomic operation
2. **Batch vector indexing** - Add vectors in bulk to HNSW
3. **Parallel processing** - Where safe (e.g., embedding validation)
4. **Progress callback** - Optional for large batches

```python
def add_batch(
    self,
    memories: List[MemoryInput],
    dedup: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,  # (completed, total)
) -> BatchResult: ...
```

#### Performance Target

| Operation | Single-item × N | Batch |
|-----------|-----------------|-------|
| 1,000 memories | ~5 seconds | <500ms |
| 10,000 memories | ~50 seconds | <3 seconds |
| 100,000 memories | ~8 minutes | <30 seconds |

#### Complementary: Batch Delete

```python
memory.delete_batch(ids: List[str]) -> BatchDeleteResult
```

Useful for:
- Bulk cleanup
- Removing all memories from a source
- Namespace deletion (see P1)

---

## P1: Good to Have Soon

These features significantly improve real-world utility but aren't strict blockers for initial adoption.

---

### Feature: Namespaces / Scoping

**Priority:** P1 - Important  
**Effort:** Medium (3-4 days)  
**Affects:** Storage layer, all query paths, API

#### Problem Statement

Single memory space doesn't work for:
- Multi-user applications (user A shouldn't see user B's memories)
- Multi-context (work vs personal)
- Multi-tenant platforms
- Testing isolation

Without namespaces, developers must:
- Create separate files per user (file management overhead)
- Manually prefix all operations (error-prone)
- Build isolation logic themselves

#### Proposed Solution

**Namespace as first-class concept:**

```python
# Option A: Parameter on every operation
memory.add(content, embedding, namespace="user_123")
memory.search(query, embedding, namespace="user_123")
memory.delete(id, namespace="user_123")

# Option B: Scoped view (cleaner for application code)
user_memory = memory.scope("user_123")
user_memory.add(content, embedding)      # Automatically scoped
user_memory.search(query, embedding)     # Only searches this namespace
user_memory.delete(id)                   # Only deletes if in this namespace
```

**Recommendation:** Support both patterns.

#### Namespace Semantics

| Behavior | Description |
|----------|-------------|
| Isolation | Queries only return results from same namespace |
| ID uniqueness | IDs are unique within namespace, not globally |
| Cross-namespace | Explicit API for cross-namespace operations (admin) |
| Default namespace | Empty string = default namespace |
| Nested namespaces | Support `org_1/user_123` style hierarchies |

#### API Design

```python
class Memory:
    def add(
        self,
        content: str,
        embedding: List[float],
        namespace: str = "",           # NEW
        ...
    ) -> str: ...
    
    def search(
        self,
        query: str,
        embedding: List[float],
        namespace: str = "",           # NEW
        ...
    ) -> List[SearchResult]: ...
    
    def scope(self, namespace: str) -> ScopedMemory:  # NEW
        """Return a view scoped to a specific namespace."""
        ...
    
    def list_namespaces(self) -> List[str]:  # NEW
        """List all namespaces in this memory file."""
        ...
    
    def delete_namespace(self, namespace: str) -> int:  # NEW
        """Delete all memories in a namespace. Returns count deleted."""
        ...


class ScopedMemory:
    """A memory view scoped to a specific namespace."""
    
    def add(self, content, embedding, ...) -> str: ...
    def search(self, query, embedding, ...) -> List[SearchResult]: ...
    def update(self, id, content, embedding) -> None: ...
    def delete(self, id) -> bool: ...
    # All operations automatically scoped
```

#### Storage Implementation

**Option A: Prefixed keys**
```
memory_id = f"{namespace}:{uuid}"
```
Simple, but namespace becomes part of ID.

**Option B: Composite keys in storage**
```
Table: memories
  namespace: String (indexed)
  id: String
  content: ...
  
Primary key: (namespace, id)
```
Cleaner, allows same ID in different namespaces.

**Recommendation:** Option B - cleaner semantics.

#### Vector Index Considerations

HNSW doesn't natively support filtering. Options:

1. **Separate index per namespace** - Clean isolation, but many small indexes
2. **Single index + post-filter** - Search globally, filter results
3. **Metadata filtering** - If usearch supports it

For v1: Start with post-filtering (simpler), optimize later if needed.

#### Use Cases

```python
# Multi-user application
for user in users:
    user_mem = memory.scope(f"user_{user.id}")
    user_mem.add(user_preference, embedding)

# Multi-context
work = memory.scope("context_work")
personal = memory.scope("context_personal")

work.add("Meeting with Bob at 3pm", embedding)
personal.add("Pick up groceries", embedding)

# Cleanup
memory.delete_namespace("user_123")  # User deleted account
```

---

### Feature: Memory Metadata Indexing

**Priority:** P1 - Important  
**Effort:** Medium (2-3 days)  
**Affects:** Storage layer, query API

#### Problem Statement

Currently metadata is stored but not queryable:

```python
memory.add(
    content="...",
    embedding=embedding,
    metadata={"type": "preference", "category": "food", "confidence": 0.9}
)

# Can't do this:
memory.search(query, embedding, filter={"type": "preference"})  # Not supported
```

Real-world needs:
- "Only retrieve facts, not preferences"
- "Only high-confidence memories"
- "Only from this session"
- "Only this category"

#### Proposed Solution

**Filterable metadata fields:**

```python
# At add time, specify indexed fields
memory.add(
    content="Prefers Italian food",
    embedding=embedding,
    metadata={
        "type": "preference",        # Will be indexed
        "category": "food",          # Will be indexed
        "confidence": 0.9,           # Will be indexed
        "raw_notes": "User mentioned..." # NOT indexed (too large/variable)
    }
)

# At query time, filter by indexed fields
results = memory.search(
    query="food preferences",
    embedding=embedding,
    filters={
        "type": "preference",
        "confidence": {"$gte": 0.8}
    }
)
```

#### Filter Syntax

Support basic operators:

```python
filters = {
    "type": "preference",              # Exact match
    "confidence": {"$gte": 0.8},       # Greater than or equal
    "category": {"$in": ["food", "travel"]},  # In list
    "deprecated": {"$ne": True},       # Not equal
}
```

#### Implementation

1. **Declare indexed fields** at memory creation or via config:
```python
memory = Memory(
    path="./data.mfdb",
    indexed_metadata=["type", "category", "confidence", "session_id"]
)
```

2. **Build secondary indexes** for declared fields

3. **Apply filters** before or after vector search depending on selectivity

#### Integration with Namespaces

Filters and namespaces compose:

```python
user_memory = memory.scope("user_123")
results = user_memory.search(
    query="food",
    embedding=embedding,
    filters={"type": "preference"}
)
# Scoped to user_123 AND filtered by type
```

---

## P2: Good to Have Later

These features are valuable but can wait for post-launch iterations.

---

### Feature: Memory Versioning / History

**Priority:** P2 - Nice to have  
**Effort:** High (5+ days)  
**Affects:** Storage model, significant complexity

#### Problem Statement

When memories are updated, the old version is lost:

```python
memory.add("Budget is $5,000", embedding, key="budget")
# Later...
memory.upsert("budget", "Budget is $7,000", new_embedding)
# $5,000 is gone forever
```

Sometimes you want:
- History of changes
- Ability to see "what did we know at time T?"
- Audit trail
- Undo capability

#### Proposed Solution (Future)

```python
# Enable versioning
memory = Memory(path="./data.mfdb", versioning=True)

# Updates create versions, not replacements
memory.upsert("budget", "Budget is $7,000", embedding)

# Query history
history = memory.get_history("budget")
# Returns: [
#   {version: 1, content: "Budget is $5,000", timestamp: "..."},
#   {version: 2, content: "Budget is $7,000", timestamp: "..."},
# ]

# Point-in-time query
results = memory.search(query, embedding, as_of="2025-01-15T10:00:00Z")
```

#### Why P2?

- Significant storage overhead
- Complex query semantics
- Most applications don't need it initially
- Can be simulated by never deleting (just marking deprecated)

---

### Feature: Memory Expiration / TTL

**Priority:** P2 - Nice to have  
**Effort:** Medium (3-4 days)  
**Affects:** Storage layer, background processes

#### Problem Statement

Some memories should automatically expire:
- Session-specific context
- Temporary preferences
- Time-limited information

#### Proposed Solution (Future)

```python
memory.add(
    content="User is currently in a meeting",
    embedding=embedding,
    ttl=3600,  # Expires in 1 hour
)

# Or explicit expiration
memory.add(
    content="Sale ends tomorrow",
    embedding=embedding,
    expires_at="2025-01-20T00:00:00Z"
)
```

#### Implementation Considerations

- Background cleanup process (or lazy cleanup on read)
- Don't return expired memories in search
- Configurable cleanup interval
- Soft delete vs hard delete

#### Why P2?

- Temporal decay already deprioritizes old memories
- Application can implement explicit deletion
- Adds background process complexity

---

### Feature: Memory Relationships (Beyond Causal)

**Priority:** P2 - Nice to have  
**Effort:** Medium (3-4 days)  
**Affects:** Graph layer, API

#### Problem Statement

Current design has:
- Causal graph (cause → effect)
- Entity graph (memory ↔ entity)

But applications might want arbitrary relationships:
- "contradicts"
- "supports"
- "supersedes"
- "relates_to"
- Custom relationship types

#### Proposed Solution (Future)

```python
# Generic relationship API
memory.add_relationship(
    from_id="mem_123",
    to_id="mem_456",
    relationship="contradicts",
    metadata={"confidence": 0.8}
)

# Query by relationship
contradictions = memory.get_related("mem_123", relationship="contradicts")

# Traverse relationships in search
results = memory.search(
    query=query,
    embedding=embedding,
    include_related=["supports", "supersedes"],  # Also return related
)
```

#### Why P2?

- Causal + Entity covers most needs
- Generic relationships add complexity
- Can be added incrementally later
- Applications can use metadata as workaround

---

### Feature: Import / Export

**Priority:** P2 - Nice to have  
**Effort:** Low-Medium (2-3 days)  
**Affects:** API, file format

#### Problem Statement

Need to:
- Migrate between MnemeFusion versions
- Debug by inspecting contents
- Transfer between systems
- Backup in human-readable format

#### Proposed Solution (Future)

```python
# Export to portable format
memory.export("backup.json")  # or .jsonl, .parquet

# Import from portable format
memory.import_from("backup.json", mode="merge")  # or "replace"

# Export specific namespace
memory.export("user_123.json", namespace="user_123")
```

#### Export Format

```json
{
  "version": "1.0",
  "exported_at": "2025-01-15T10:00:00Z",
  "memories": [
    {
      "id": "mem_abc123",
      "namespace": "default",
      "content": "Budget is $5,000",
      "embedding": [0.1, 0.2, ...],
      "metadata": {"type": "fact"},
      "source": {"type": "conversation", "id": "..."},
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "causal_edges": [
    {"from": "mem_123", "to": "mem_456"}
  ],
  "entity_edges": [
    {"memory": "mem_123", "entity": "budget"}
  ]
}
```

#### Why P2?

- Binary format works for normal operations
- Can inspect with custom tooling
- Migration can be handled at application level initially

---

## Summary Matrix

| Feature | Priority | Effort | Phase | Core Benefit |
|---------|----------|--------|-------|--------------|
| **Deduplication / Upsert** | P0 | Medium | 1 | Prevents pollution |
| **Provenance / Source** | P0 | Low | 1 | Explainability |
| **Batch Operations** | P0 | Low | 1 | Performance |
| **Namespaces / Scoping** | P1 | Medium | 2 | Multi-tenant |
| **Metadata Indexing** | P1 | Medium | 2 | Filtered retrieval |
| Memory Versioning | P2 | High | Post | Audit trail |
| Memory TTL | P2 | Medium | Post | Auto-cleanup |
| Generic Relationships | P2 | Medium | Post | Flexibility |
| Import / Export | P2 | Low | Post | Portability |

---

## Implementation Order Recommendation

### Phase 1 (v1.0)

1. **Provenance** - Low effort, high value, do first
2. **Batch Operations** - Low effort, essential for adoption
3. **Deduplication** - Medium effort, prevents early pollution

### Phase 2 (v1.1)

4. **Namespaces** - Unlocks multi-tenant use cases
5. **Metadata Indexing** - Enables filtered retrieval

### Post-Launch (v1.x)

6. Import/Export - Developer convenience
7. TTL - Nice to have
8. Versioning - If demand exists
9. Generic Relationships - If demand exists

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-01-17 | Initial feature specifications |
