# MnemeFusion: Implementation Plan

**Document Version:** 1.0
**Created:** January 2026
**Status:** Active Development Plan

---

## Table of Contents

1. [Overview](#overview)
2. [Sprint Structure](#sprint-structure)
3. [Phase 1: Core Engine (Sprints 1-8)](#phase-1-core-engine)
4. [Phase 2: Production Hardening (Sprints 9-14)](#phase-2-production-hardening)
5. [Phase 3: Ecosystem (Sprints 15+)](#phase-3-ecosystem)
6. [Risk Management](#risk-management)
7. [Success Criteria](#success-criteria)

---

## Overview

### Timeline Summary

| Phase | Duration | Sprints | Focus |
|-------|----------|---------|-------|
| Phase 1 | 4 months | 1-8 | Core engine with 4D indexing |
| Phase 2 | 3 months | 9-14 | Production hardening & optimization |
| Phase 3 | Ongoing | 15+ | Ecosystem & community |

### Development Principles

1. **Compose, don't reimplement** - Use proven libraries
2. **Test early, test often** - TDD approach where practical
3. **Document as you build** - Keep docs in sync with code
4. **Benchmark continuously** - Performance tracking from day one
5. **Python-first** - Ensure Python API is ergonomic

---

## Sprint Structure

Each sprint is 2 weeks with:
- **Planning** (Day 1): Story selection, task breakdown
- **Development** (Days 2-9): Implementation & testing
- **Review** (Day 10): Demo, retrospective, planning next sprint

**Sprint Capacity**: ~60-80 hours per sprint (15-20 hours/week)

---

## Phase 1: Core Engine

**Goal**: Working engine with all four dimensions, Python bindings, basic query planner

### Sprint 1: Project Foundation (Weeks 1-2) ✅ COMPLETE

**Objective**: Set up Rust project structure, storage layer, and development environment

**Completion Date**: January 14, 2026

#### Stories

**[STORY-1.1] As a developer, I can create and open a MnemeFusion database file** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Can create new `.mfdb` file
  - ✅ Can open existing `.mfdb` file
  - ✅ File header with magic number and version validation
  - ✅ Basic error handling for corrupt files
  - ✅ Integration test demonstrating create/open/close

**[STORY-1.2] As a developer, I can store and retrieve memory records** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Memory struct with id, content, timestamp, metadata
  - ✅ Store memory to redb tables
  - ✅ Retrieve memory by ID
  - ✅ Delete memory by ID
  - ✅ Unit tests for all CRUD operations

#### Tasks

**Setup & Infrastructure**
- [x] Create Cargo workspace with `mnemefusion-core` crate
- [x] Add dependencies: redb, thiserror, uuid, serde_json
- [x] Configure Rust toolchain (1.75+)
- [x] Set up testing framework and conventions
- [x] Create `.gitignore` and CI/CD skeleton

**Core Types** (types/)
- [x] Implement `MemoryId` (UUID-based, u64 conversion for usearch)
- [x] Implement `Memory` struct with serialization
- [x] Implement `Timestamp` utilities
- [x] Implement `Error` enum with thiserror
- [x] Implement `Config` struct with defaults
- [x] Add comprehensive unit tests for all types

**Storage Layer** (storage/)
- [x] Implement `FileHeader` with validation
- [x] Implement `StorageEngine` wrapper around redb
- [x] Define table schemas (MEMORIES, TEMPORAL_INDEX, etc.)
- [x] Implement `store_memory()` / `get_memory()` / `delete_memory()`
- [x] Implement transaction helper methods
- [x] Write integration tests with tempdir

**Top-Level API** (lib.rs, memory.rs)
- [x] Create `MemoryEngine` struct (main API entry point)
- [x] Implement `MemoryEngine::open(path, config)`
- [x] Implement `MemoryEngine::close()`
- [x] Write basic integration test: create, add memory, retrieve, close
- [x] Document public APIs with rustdoc

**Documentation**
- [x] Update README with build instructions
- [x] Document module structure in CLAUDE.md
- [x] Write examples/basic_usage.rs

**Sprint 1 Review** ✅
- ✅ Can create/open database
- ✅ Can store/retrieve memories
- ✅ All tests passing (50 unit + 6 integration + 7 doc = 63 total)
- ✅ Basic documentation complete
- ✅ Example code working
- ✅ Performance targets met (add: ~1ms, get: ~0.1ms)

---

### Sprint 2: Vector Index Integration (Weeks 3-4) ✅ COMPLETE

**Objective**: Integrate usearch for semantic similarity, add memories with embeddings

**Completion Date**: January 18, 2026

#### Stories

**[STORY-2.1] As a developer, I can add memories with vector embeddings** ✅
- **Priority**: P0 (Critical)
- **Points**: 13
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Add memory with content + embedding vector
  - ✅ Vector automatically indexed in usearch
  - ✅ Embedding dimension validation
  - ✅ Persist vector index to storage
  - ✅ Load vector index on database open
  - ✅ Integration test with 1000+ memories

**[STORY-2.2] As a developer, I can search memories by semantic similarity** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Search by query embedding, return top-k results
  - ✅ Results include memory ID and similarity score
  - ✅ Similarity scores normalized (0.0-1.0)
  - ✅ Handles empty index gracefully
  - ✅ Performance: <5ms for 1K memories (exceeded target)

#### Tasks

**Library Evaluation**
- [x] Benchmark usearch vs hora (HNSW performance) - Chose usearch
- [x] Test usearch persistence API - Working with save_to_buffer/load_from_buffer
- [x] Validate usearch Python bindings compatibility - Compatible
- [x] Document choice in decision log - Documented in PROJECT_STATE.md

**Vector Index** (index/vector.rs)
- [x] Wrap usearch Index with VectorIndex struct
- [x] Implement `VectorIndex::new(dimension, storage)` with reserve(1000) for Windows
- [x] Implement `add(id, embedding)` with dimension validation
- [x] Implement `search(query, top_k)` returning VectorResult
- [x] Implement `remove(id)` for deletions
- [x] Implement `save()` - serialize index with adaptive buffer sizing
- [x] Implement `load()` - deserialize index from storage
- [x] Handle usearch errors, convert to Error enum
- [x] Write unit tests with small test index (8 tests)

**Integration with MemoryEngine**
- [x] Add VectorIndex to MemoryEngine struct (with Arc<RwLock>)
- [x] Update `add()` to accept embedding parameter
- [x] Auto-index embedding when adding memory
- [x] Persist vector index on close()
- [x] Load vector index on open()
- [x] Update Config with embedding_dim parameter

**Storage Enhancement**
- [x] Add MEMORY_ID_INDEX table for u64 → MemoryId reverse lookup
- [x] Implement `get_memory_by_u64()` for efficient search result retrieval
- [x] Update `store_memory()` to maintain reverse index
- [x] Update `delete_memory()` to clean up reverse index

**Testing & Benchmarking**
- [x] Integration test: add 1000 memories, search, verify results
- [x] Benchmark: add performance (~2ms per memory, exceeds target)
- [x] Benchmark: search performance (<5ms for 1K memories, exceeds target)
- [x] Test index persistence and reload
- [x] Test dimension mismatch error handling

**Documentation**
- [x] Document VectorIndex API with rustdoc
- [x] Update examples/basic_usage.rs with search demonstration
- [x] Document performance in PROJECT_STATE.md

**Sprint 2 Review**
- ✅ Vector indexing working
- ✅ Semantic search returns ranked results
- ✅ Index persists and reloads correctly
- ✅ Performance benchmarks meet targets

---

### Sprint 3: Temporal Index (Weeks 5-6) ✅ COMPLETE

**Objective**: Implement temporal indexing and time-based queries

**Completion Date**: January 19, 2026

#### Stories

**[STORY-3.1] As a developer, I can query memories by time range** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Query memories within start-end timestamp range
  - ✅ Results sorted by timestamp (newest first)
  - ✅ Efficient B-tree range queries via redb
  - ✅ Support "most recent N" queries
  - ✅ Handle edge cases (empty range, future timestamps)

**[STORY-3.2] As a developer, I can assign custom timestamps to memories** ✅
- **Priority**: P1 (High)
- **Points**: 5
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Optional timestamp parameter in add()
  - ✅ Defaults to current time if not provided
  - ✅ Store timestamp in memory record
  - ✅ Timestamp automatically indexed

#### Tasks

**Temporal Index** (index/temporal.rs)
- [x] Implement TemporalIndex struct
- [x] Implement `range_query(start, end, limit)` - returns TemporalResult (newest first)
- [x] Implement `recent(n)` - returns N most recent memories (reverse iterator)
- [x] Implement `count_range()` - efficient counting without loading records
- [x] Use redb native ordering (timestamp as key)
- [x] Write unit tests with various time ranges (8 tests)

**Timestamp Utilities** (types/timestamp.rs)
- [x] All timestamp helper methods already existed from Sprint 1:
  - `now()` ✅
  - `from_unix_secs()` / `as_unix_secs()` ✅
  - `subtract_days(n)` / `add_days(n)` ✅
  - `start_of_day()` / `end_of_day()` ✅
  - `as_micros()` (for redb key) ✅
- [x] All utilities already had unit tests from Sprint 1 ✅

**Integration**
- [x] Add TemporalIndex to MemoryEngine (Arc<TemporalIndex>)
- [x] Timestamp indexing already happening from Sprint 1 (store_memory) ✅
- [x] Add `get_recent(n)` method to MemoryEngine
- [x] Add `get_range(start, end, limit)` method to MemoryEngine
- [x] Added db() accessor to StorageEngine for index access
- [x] Made TEMPORAL_INDEX accessible to temporal module (pub(crate))

**Testing**
- [x] Integration test: add memories with various timestamps
- [x] Test range queries across different periods
- [x] Test recent() with various limits
- [x] Test ordering (newest first)
- [x] Test combined temporal + semantic search
- [x] Test edge cases (empty range, future timestamps, limits)
- [x] All 87 tests passing (66 unit + 9 integration + 12 doc)

**Documentation**
- [x] Document temporal queries in API docs (rustdoc)
- [x] Add temporal query examples to basic_usage.rs
- [x] Updated PROJECT_STATE.md and IMPLEMENTATION_PLAN.md

**Sprint 3 Review**
- ✅ Temporal queries working
- ✅ Range queries efficient
- ✅ Custom timestamps supported
- ✅ All tests passing

---

### Sprint 4: Causal Graph Foundation (Weeks 7-8) ✅ COMPLETE

**Objective**: Implement causal graph structure and persistence

**Completion Date**: January 19, 2026

#### Stories

**[STORY-4.1] As a developer, I can link memories with causal relationships** ✅
- **Priority**: P0 (Critical)
- **Points**: 13
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Add causal link between two memory IDs
  - ✅ Links have confidence score (0.0-1.0)
  - ✅ Links have evidence text
  - ✅ Bidirectional indexes (forward and reverse via petgraph)
  - ✅ Persist causal edges to storage

**[STORY-4.2] As a developer, I can query causal chains** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Get causes of a memory (backward traversal)
  - ✅ Get effects of a memory (forward traversal)
  - ✅ Multi-hop traversal with max_hops limit
  - ✅ Return path and cumulative confidence
  - ✅ Handle cycles gracefully

#### Tasks

**Graph Structures** (graph/causal.rs)
- [x] Define CausalEdge struct (confidence, evidence)
- [x] Implement GraphManager with petgraph DiGraph
- [x] Implement `add_causal_link(cause, effect, confidence, evidence)`
- [x] Build memory_id → NodeIndex HashMap
- [x] Write unit tests for graph construction (11 unit tests)

**Graph Traversal**
- [x] Implement `get_causes(memory_id, max_hops)` - BFS backward
- [x] Implement `get_effects(memory_id, max_hops)` - BFS forward
- [x] Calculate cumulative confidence along paths
- [x] Return CausalTraversalResult with path info
- [x] Handle disconnected nodes (returns error)
- [x] Detect and prevent infinite loops (visited tracking)
- [x] Write unit tests with sample graphs (comprehensive test coverage)

**Graph Persistence** (graph/persist.rs)
- [x] Implement `save_causal_graph()` - serialize to redb
- [x] Store edges in CAUSAL_GRAPH table (single table for edge list)
- [x] Implement `load_causal_graph()` - reconstruct from redb
- [x] Mark graph as dirty on mutations (implicit - always saves on close)
- [x] Auto-save on close() (integrated with MemoryEngine::close)
- [x] Test persistence round-trip (3 persist tests)

**Integration**
- [x] Add GraphManager to MemoryEngine (Arc<RwLock<GraphManager>>)
- [x] Expose `add_causal_link()` on MemoryEngine
- [x] Expose `get_causes()` and `get_effects()`
- [x] Load graph on open(), save on close()

**Testing**
- [x] Unit tests: simple 2-node graphs
- [x] Unit tests: multi-hop chains (A→B→C→D)
- [x] Unit tests: branching graphs (A→B, A→C)
- [x] Integration test: create graph, save, reload, verify (test_causal_graph_persistence)
- [x] Test max_hops limiting (test_max_hops_limit)
- [x] Performance: efficient traversal with BFS

**Documentation**
- [x] Document causal graph API (rustdoc on all public methods)
- [x] Add causal query examples (basic_usage.rs updated)
- [ ] Diagram causal graph structure in architecture docs (deferred)

**Sprint 4 Review**
- ✅ Causal links working
- ✅ Multi-hop traversal functional
- ✅ Graph persists and reloads
- ✅ Performance acceptable

---

### Sprint 5: Entity Graph Foundation (Weeks 9-10) ✅ COMPLETE

**Objective**: Implement entity extraction and entity-memory graph

**Completion Date**: January 20, 2026

#### Stories

**[STORY-5.1] As a developer, I can create and track entities** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Create Entity with ID and name
  - ✅ Store entities in ENTITIES table
  - ✅ Index by name in ENTITY_NAMES table (case-insensitive)
  - ✅ Find entity by name (case-insensitive)
  - ✅ Update and delete entities
  - ✅ Entity mention counting

**[STORY-5.2] As a developer, I can link memories to entities** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Link memory to entity with relationship type
  - ✅ Relationship type: "mentions" (single type for MVP)
  - ✅ Store links in entity graph (petgraph bipartite)
  - ✅ Query all memories for an entity
  - ✅ Query all entities for a memory
  - ✅ Entity graph persistence (save/load)

**[STORY-5.3] As a developer, I can extract entities from memory content (basic)** ✅
- **Priority**: P2 (Medium)
- **Points**: 5
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Simple entity extraction (capitalized words)
  - ✅ Automatic entity linking on add()
  - ✅ Configurable: enable/disable auto-extraction
  - ✅ Handles single and multi-word entities
  - ✅ Filters common stop words
  - Note: Advanced NER deferred to Phase 2

#### Tasks

**Entity Types** (types/entity.rs)
- [x] Define Entity struct (id, name, metadata, mention_count)
- [x] Define EntityId type
- [x] Implement serialization (serde)
- [x] Write unit tests (8 tests)

**Entity Graph** (graph/entity.rs)
- [x] Define EntityNode enum (Entity | Memory)
- [x] Define EntityEdge struct (relationship: "mentions")
- [x] Add entity_graph DiGraph to GraphManager
- [x] Implement `link_memory_to_entity(memory_id, entity_id)`
- [x] Implement `get_entity_memories(entity_id)` - returns EntityQueryResult
- [x] Implement `get_memory_entities(memory_id)` - returns Vec<EntityId>
- [x] Write unit tests (8 tests)

**Entity Storage**
- [x] Implement entity storage in StorageEngine (ENTITIES, ENTITY_NAMES tables)
- [x] `store_entity()` / `get_entity()` / `find_entity_by_name()`
- [x] Case-insensitive name lookup via ENTITY_NAMES index
- [x] Persist entity graph edges in METADATA_TABLE
- [x] Load entity graph on open() via persist module

**Basic Entity Extraction** (ingest/entity_extractor.rs)
- [x] Define EntityExtractor trait
- [x] Implement SimpleEntityExtractor (capitalized words)
- [x] Filter common stop words (days, months, articles)
- [x] Return Vec<String> (entity names)
- [x] Make pluggable for future advanced extractors (trait design)
- [x] Write unit tests (8 tests)

**Integration**
- [x] Add entity operations to MemoryEngine API
- [x] Auto-extract entities in `add()` if enabled (config.entity_extraction_enabled)
- [x] Expose `get_entity_memories()`, `get_memory_entities()`, `list_entities()`
- [x] Entity extraction flag already in Config (entity_extraction_enabled)
- [x] Update basic_usage example with entity queries

**Testing**
- [x] Unit tests for entity storage (entity types: 8 tests)
- [x] Unit tests for entity graph operations (entity graph: 8 tests)
- [x] Unit tests for entity extraction (extractor: 8 tests)
- [x] Integration via basic_usage example (demonstrates end-to-end workflow)
- [x] Test case-insensitive entity lookup
- [x] Test entity extraction on sample texts

**Documentation**
- [x] Document Entity API (rustdoc on all public methods)
- [x] Add entity examples (basic_usage.rs updated with entity demonstrations)
- [x] Document entity extraction limitations (noted in CLAUDE.md and code comments)

**Sprint 5 Review**
- ✅ Entity creation and storage working
- ✅ Entity-memory links functional
- ✅ Basic entity extraction operational
- ✅ Tests passing

---

### Sprint 6: Ingestion Pipeline (Weeks 11-12) ✅ COMPLETE

**Objective**: Unify memory ingestion across all dimensions

#### Stories

**[STORY-6.1] As a developer, I can add a memory and have all dimensions automatically indexed** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Single `add()` call indexes all dimensions
  - ✅ Atomic transaction (all or nothing)
  - ✅ Rollback on any index failure
  - ⏸️ Efficient batch operations (deferred to later sprint)
  - ✅ Performance: <15ms per memory for all dimensions

**[STORY-6.2] As a developer, I can delete a memory and clean up all indexes** ✅
- **Priority**: P1 (High)
- **Points**: 5
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Remove from all indexes
  - ✅ Clean up orphaned entities
  - ✅ Clean up causal links
  - ✅ Atomic deletion

**[STORY-6.3] Transaction coordination and rollback** ✅
- **Priority**: P0 (Critical)
- **Points**: 8
- **Status**: COMPLETE
- **Acceptance Criteria**:
  - ✅ Rollback on failure
  - ✅ Prevent partial state
  - ✅ Entity deduplication within memory

#### Tasks

**Ingestion Pipeline** (ingest/pipeline.rs)
- [x] Create IngestionPipeline struct
- [x] Aggregate references to all indexes
- [x] Implement unified `add()` method:
  - [x] Validate inputs (done in MemoryEngine)
  - [x] Create Memory record
  - [x] Store in storage
  - [x] Add to vector index
  - [x] Add to temporal index
  - [x] Extract and link entities
  - [x] Deduplicate entities within memory
- [x] Implement transaction coordinator (manual rollback)
- [x] Handle partial failure (rollback)
- [x] Write unit tests (8 tests)

**Deletion Pipeline**
- [x] Implement `delete(memory_id)` method:
  - [x] Remove from storage
  - [x] Remove from vector index
  - [x] Remove from temporal index
  - [x] Remove causal links (via remove_memory_from_causal_graph)
  - [x] Remove entity links (via remove_memory_from_entity_graph)
  - [x] Clean up orphaned entities (mention_count = 0)
- [x] Test cascading deletes
- [x] Test atomic rollback on failure
- [x] Fix petgraph NodeIndex invalidation bug

**Batch Operations** (deferred to later sprint)
- [ ] Implement `add_batch(Vec<Memory>)` for efficiency
- [ ] Batch insert to indexes
- [ ] Single transaction for batch
- [ ] Performance: <10ms per memory in batch

**Integration**
- [x] Refactor MemoryEngine to use IngestionPipeline
- [x] Ensure all `add()` calls route through pipeline
- [x] Add temporal index add/remove methods
- [x] Add causal graph remove_memory method
- [x] Fix entity graph rebuild_node_maps bug

**Testing**
- [x] Integration test: add memory, verify all indexes updated
- [x] Test rollback on index failure (wrong embedding dimension)
- [x] Test delete with cascading cleanup
- [x] Test orphaned entity cleanup
- [x] Test entity deduplication
- [ ] Performance test: add 10K memories (deferred)
- [ ] Stress test: concurrent adds (deferred)

**Documentation**
- [x] Document ingestion flow in code comments
- [ ] Add diagrams for data flow (deferred)
- [x] Document transaction guarantees in code comments

**Sprint 6 Review**
- ✅ Unified ingestion working
- ✅ All dimensions indexed atomically
- ✅ Deletion cleans up properly
- ✅ Performance targets met (<10ms per memory)
- ✅ 110 tests passing (8 new pipeline tests)
- ✅ Fixed critical petgraph bug

---

### Sprint 7: Query Planner & Intent Classification (Weeks 13-14)

**Objective**: Implement intent-aware query planning

#### Stories

**[STORY-7.1] As a developer, I can classify query intent from natural language**
- **Priority**: P0 (Critical)
- **Points**: 8
- **Acceptance Criteria**:
  - Classify into: Factual, Temporal, Causal, Entity, MultiHop, Open
  - Return intent + confidence score
  - Pattern-based (regex) classification
  - Handle ambiguous queries
  - 70%+ accuracy on sample queries

**[STORY-7.2] As a developer, I can create adaptive query plans based on intent**
- **Priority**: P0 (Critical)
- **Points**: 8
- **Acceptance Criteria**:
  - Map intent to weight distribution
  - Extract query signals (time range, entities, keywords)
  - Determine execution order
  - Return QueryPlan struct
  - Configurable weight profiles

#### Tasks

**Intent Classification** (query/intent.rs)
- [ ] Define QueryIntent enum
- [ ] Implement IntentClassifier with regex patterns:
  - Temporal: "when", "yesterday", "last week"
  - Causal: "why", "because", "caused"
  - Entity: "who", "tell me about"
  - MultiHop: "compare", "difference"
- [ ] Implement `classify(query)` returning (Intent, f32)
- [ ] Implement signal extraction:
  - `extract_temporal_range()`
  - `extract_entity_mentions()`
  - `extract_keywords()`
- [ ] Write unit tests with 50+ sample queries
- [ ] Measure classification accuracy

**Query Planner** (query/planner.rs)
- [ ] Define QueryPlan struct
- [ ] Define AdaptiveWeightConfig with default weights:
  - Factual: semantic=0.6, temporal=0.1, causal=0.1, entity=0.2
  - Temporal: semantic=0.2, temporal=0.6, causal=0.1, entity=0.1
  - Causal: semantic=0.2, temporal=0.1, causal=0.6, entity=0.1
  - Entity: semantic=0.2, temporal=0.1, causal=0.1, entity=0.6
- [ ] Implement QueryPlanner::plan(query)
- [ ] Determine execution order based on intent
- [ ] Write unit tests with various query types

**Integration**
- [ ] Add QueryPlanner to MemoryEngine
- [ ] Update `search()` to use planner:
  - Classify intent
  - Get query plan
  - Execute searches according to plan
  - Pass weights to fusion
- [ ] Make weights configurable in Config

**Testing**
- [ ] Unit tests for intent classification
- [ ] Unit tests for query planning
- [ ] Integration test: query with different intents
- [ ] Validate execution order matches intent
- [ ] Test edge cases (empty query, gibberish)

**Documentation**
- [ ] Document QueryIntent types
- [ ] Document weight configurations
- [ ] Add query planning examples
- [ ] Explain intent classification logic

**Sprint 7 Review**
- ✅ Intent classification working
- ✅ Query plans generated correctly
- ✅ Weights adjust based on intent
- ✅ Tests passing

---

### Sprint 8: Fusion Engine & Python Bindings (Weeks 15-16)

**Objective**: Complete fusion algorithm and expose Python API

#### Stories

**[STORY-8.1] As a developer, I can fuse results from all dimensions with adaptive weighting**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - Combine semantic, temporal, causal, entity results
  - Apply adaptive weights from query plan
  - Normalize scores across dimensions
  - Rank by fused score
  - Return top-k results with score breakdown

**[STORY-8.2] As a Python developer, I can use MnemeFusion from Python**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - `pip install mnemefusion` works (local dev install)
  - Python API mirrors Rust API
  - Type hints and docstrings
  - Error handling with Python exceptions
  - Example Python scripts work

#### Tasks

**Fusion Engine** (query/fusion.rs)
- [ ] Define FusionEngine struct
- [ ] Define FusedResult struct with score breakdown
- [ ] Implement temporal decay function (exponential decay)
- [ ] Implement causal scoring (hop penalty + confidence)
- [ ] Implement `fuse()` method:
  - Collect all candidate memory IDs
  - Calculate per-dimension scores
  - Apply adaptive weights
  - Normalize for active dimensions only
  - Calculate fused score
  - Sort by fused score descending
- [ ] Write unit tests with synthetic data

**Search Implementation**
- [ ] Implement full `search()` in MemoryEngine:
  - Plan query
  - Execute vector search
  - Execute temporal query (with decay)
  - Execute causal traversal (if relevant)
  - Execute entity lookup (if relevant)
  - Fuse all results
  - Retrieve full Memory objects for top-k
  - Return SearchResult structs
- [ ] Optimize for common cases (factual = mostly semantic)
- [ ] Add search result caching (optional)

**Python Bindings** (mnemefusion-python/)
- [ ] Set up PyO3 project with maturin
- [ ] Create pyproject.toml with package metadata
- [ ] Implement Python wrapper types:
  - Memory (main class)
  - Config
  - SearchResult
  - MemoryItem
- [ ] Implement Python methods:
  - `__init__(path, config=None)`
  - `add(content, embedding, metadata=None, timestamp=None)`
  - `search(query, embedding, top_k=10)`
  - `get(id)`
  - `add_causal_link(cause_id, effect_id, confidence, evidence)`
  - `get_causes(id, max_hops=2)`
  - `get_effects(id, max_hops=2)`
  - `get_entity_memories(entity_name)`
  - `close()`
- [ ] Add Python docstrings with examples
- [ ] Add type hints (stubs file)
- [ ] Error conversion: Rust Error → PyValueError

**Testing**
- [ ] Unit tests for fusion algorithm
- [ ] Unit tests for temporal decay
- [ ] Integration test: full search flow
- [ ] Python tests with pytest:
  - Test create/open database
  - Test add memories
  - Test search with different intents
  - Test causal operations
  - Test error handling
- [ ] Test memory leaks (Python GC interaction)

**Examples & Documentation**
- [ ] Create examples/basic_usage.py
- [ ] Create examples/python/causal_chains.py
- [ ] Update README with Python quickstart
- [ ] Generate API documentation (mkdocs)

**Build & Distribution**
- [ ] Test maturin develop (local install)
- [ ] Test maturin build (wheel generation)
- [ ] Document installation process
- [ ] Test on Windows, Linux, macOS (if possible)

**Sprint 8 Review**
- ✅ Fusion algorithm working
- ✅ Python bindings functional
- ✅ Can install via maturin develop
- ✅ Example scripts run successfully
- ✅ Phase 1 complete! 🎉

---

## Phase 2: Production Hardening

**Goal**: Production-ready reliability, performance optimization, comprehensive testing

### Sprint 9: Reliability & ACID (Weeks 17-18)

**Objective**: Ensure ACID guarantees and crash recovery

#### Stories

**[STORY-9.1] As a user, my data is safe even if the process crashes**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - All writes are atomic
  - No partial state after crash
  - Database recoverable after crash
  - Redb transactions fully utilized
  - Vector index recoverable

**[STORY-9.2] As a developer, I can detect and handle corrupt databases**
- **Priority**: P1 (High)
- **Points**: 8
- **Acceptance Criteria**:
  - Validate file header on open
  - Detect truncated files
  - Detect invalid index data
  - Return clear error messages
  - Optional: repair mode

#### Tasks

**Transaction Coordination**
- [ ] Audit all write operations for transaction boundaries
- [ ] Ensure vector index saves are in transactions
- [ ] Ensure graph saves are in transactions
- [ ] Implement write-ahead logging (if needed)
- [ ] Test crash during write (kill -9)

**Corruption Detection**
- [ ] Validate file header magic number
- [ ] Check version compatibility
- [ ] Validate table schemas on open
- [ ] Validate vector index integrity
- [ ] Add checksums to critical data (optional)
- [ ] Write corruption recovery tests

**Error Handling Audit**
- [ ] Review all Error types for clarity
- [ ] Add context to error messages
- [ ] Document error recovery strategies
- [ ] Test error paths (inject failures)

**Testing**
- [ ] Chaos tests: kill process during operations
- [ ] Corruption tests: manually corrupt files
- [ ] Recovery tests: simulate power loss
- [ ] Test redb's built-in durability guarantees
- [ ] Document ACID properties in architecture

**Documentation**
- [ ] Document transaction guarantees
- [ ] Document crash recovery behavior
- [ ] Add troubleshooting guide

**Sprint 9 Review**
- ✅ ACID guarantees verified
- ✅ Crash recovery working
- ✅ Corruption detection functional

---

### Sprint 10: Performance Optimization (Weeks 19-20)

**Objective**: Optimize hot paths and meet latency targets

#### Stories

**[STORY-10.1] As a user, search latency is consistently under 10ms for 100K memories**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - p50 latency <5ms
  - p99 latency <10ms
  - No memory leaks
  - Efficient resource usage
  - Benchmark suite established

#### Tasks

**Profiling**
- [ ] Set up criterion benchmarks
- [ ] Profile add() operation
- [ ] Profile search() operation
- [ ] Profile fusion algorithm
- [ ] Identify bottlenecks with flamegraph

**Optimizations**
- [ ] Optimize MemoryId conversions (remove allocations)
- [ ] Cache frequently accessed memories
- [ ] Lazy load entity/causal graphs
- [ ] Optimize score normalization
- [ ] Use SIMD for vector operations (if applicable)
- [ ] Pool reusable buffers

**Index Tuning**
- [ ] Tune usearch HNSW parameters (M, ef_construction)
- [ ] Test different quantization options (f16, i8)
- [ ] Optimize temporal index range queries
- [ ] Index maintenance strategies

**Memory Management**
- [ ] Reduce allocations in hot paths
- [ ] Use `&str` instead of `String` where possible
- [ ] Profile memory usage with valgrind/heaptrack
- [ ] Implement memory limits (configurable)

**Benchmarking**
- [ ] Benchmark suite with 10K, 100K, 1M memories
- [ ] Add operation: measure throughput
- [ ] Search operation: measure latency (p50, p99)
- [ ] Memory footprint: measure RSS
- [ ] Compare with Qdrant/Chroma (reference)

**Documentation**
- [ ] Publish benchmark results
- [ ] Document performance characteristics
- [ ] Add optimization guide

**Sprint 10 Review**
- ✅ Latency targets met
- ✅ Memory usage acceptable
- ✅ Benchmark suite established

---

### Sprint 11: Comprehensive Testing (Weeks 21-22)

**Objective**: Achieve >80% test coverage, add property-based tests

#### Stories

**[STORY-11.1] As a developer, I have confidence in code quality through comprehensive tests**
- **Priority**: P1 (High)
- **Points**: 13
- **Acceptance Criteria**:
  - >80% line coverage
  - Property-based tests for core algorithms
  - Integration tests for all user flows
  - Performance regression tests
  - Test suite runs in CI

#### Tasks

**Test Coverage**
- [ ] Measure current coverage with tarpaulin
- [ ] Add unit tests for uncovered paths
- [ ] Add edge case tests
- [ ] Add error path tests
- [ ] Target: >80% coverage

**Property-Based Testing**
- [ ] Add proptest dependency
- [ ] Property tests for MemoryId conversions
- [ ] Property tests for fusion algorithm (weights always sum to 1)
- [ ] Property tests for timestamp utilities
- [ ] Property tests for score normalization

**Integration Tests**
- [ ] End-to-end user flows:
  - Create DB → Add memories → Search → Close → Reopen
  - Build causal graph → Query chains
  - Add entities → Query entity memories
- [ ] Multi-query test suite (50+ queries)
- [ ] Concurrent access tests (if supported)

**Regression Tests**
- [ ] Benchmark regression detection
- [ ] API compatibility tests
- [ ] File format backward compatibility

**CI/CD**
- [ ] Set up GitHub Actions
- [ ] Run tests on commit
- [ ] Run benchmarks on PR
- [ ] Generate coverage reports
- [ ] Fail on regressions

**Documentation**
- [ ] Testing guide for contributors
- [ ] CI/CD documentation

**Sprint 11 Review**
- ✅ Test coverage >80%
- ✅ Property tests passing
- ✅ CI/CD functional

---

### Sprint 12: API Stability & Documentation (Weeks 23-24)

**Objective**: Finalize API for 1.0, comprehensive documentation

#### Stories

**[STORY-12.1] As a user, the API is stable and well-documented**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - API reviewed for consistency
  - All public APIs documented
  - Python type stubs complete
  - User guide written
  - API reference generated
  - Migration guide (if APIs changed)

#### Tasks

**API Review**
- [ ] Review Rust API for consistency
- [ ] Review Python API for Pythonic idioms
- [ ] Standardize naming conventions
- [ ] Simplify where possible
- [ ] Consider future extensibility
- [ ] Document breaking changes from early sprints

**Documentation**
- [ ] Write comprehensive API reference (Rust)
- [ ] Write comprehensive API reference (Python)
- [ ] Create user guide with tutorials:
  - Getting started
  - Basic usage
  - Advanced queries
  - Causal reasoning
  - Entity management
  - Performance tuning
- [ ] Add architecture deep-dive
- [ ] Create FAQ
- [ ] Add troubleshooting guide

**Python Type Stubs**
- [ ] Generate .pyi files for type checking
- [ ] Ensure mypy compatibility
- [ ] Add to package distribution

**Examples**
- [ ] Create 5+ example applications:
  - Personal journal with memory
  - Chatbot with context
  - Research assistant
  - Meeting notes organizer
  - Code snippet manager
- [ ] Document each example

**Website (optional)**
- [ ] Create documentation website (MkDocs)
- [ ] Deploy to GitHub Pages
- [ ] Add quickstart guide
- [ ] Add API playground (optional)

**Sprint 12 Review**
- ✅ API stable
- ✅ Documentation comprehensive
- ✅ Examples working

---

### Sprint 13: Python Package Distribution (Weeks 25-26)

**Objective**: Publish Python package, automated builds

#### Stories

**[STORY-13.1] As a Python user, I can `pip install mnemefusion` without Rust**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - Pre-built wheels for Windows, Linux, macOS
  - Published to PyPI (test.pypi.org first)
  - Installation documented
  - Versioning strategy defined
  - Automated release pipeline

#### Tasks

**Build Infrastructure**
- [ ] Set up maturin with GitHub Actions
- [ ] Configure cross-compilation (cibuildwheel)
- [ ] Build wheels for:
  - Linux x86_64 (manylinux)
  - macOS x86_64 and ARM64
  - Windows x86_64
- [ ] Test wheels on each platform
- [ ] Generate sdist for source distribution

**PyPI Publishing**
- [ ] Create PyPI account
- [ ] Register package name
- [ ] Test upload to test.pypi.org
- [ ] Validate installation from test.pypi.org
- [ ] Document release process

**Versioning**
- [ ] Adopt Semantic Versioning (SemVer)
- [ ] Set initial version: 0.1.0
- [ ] Document versioning policy
- [ ] Add version to package metadata

**Automated Release**
- [ ] GitHub Actions workflow for releases
- [ ] Trigger on git tag push
- [ ] Build all wheels
- [ ] Upload to PyPI
- [ ] Create GitHub release with notes

**Installation Testing**
- [ ] Test installation on fresh systems
- [ ] Test dependency resolution
- [ ] Verify no Rust required for users
- [ ] Document system requirements

**Documentation**
- [ ] Installation guide
- [ ] Release notes template
- [ ] Changelog

**Sprint 13 Review**
- ✅ Wheels build successfully
- ✅ Package installable via pip
- ✅ Automated releases working

---

### Sprint 14: Production Readiness & Polish (Weeks 27-28)

**Objective**: Final polish, prepare for 1.0 release

#### Stories

**[STORY-14.1] As a user, I have confidence that MnemeFusion is production-ready**
- **Priority**: P0 (Critical)
- **Points**: 13
- **Acceptance Criteria**:
  - All P0 and P1 issues resolved
  - Performance benchmarks published
  - Security review complete
  - License and legal documentation complete
  - 1.0 release checklist complete

#### Tasks

**Final Testing**
- [ ] Full regression test suite
- [ ] Stress testing (millions of memories)
- [ ] Long-running stability test (24+ hours)
- [ ] Memory leak detection
- [ ] Thread safety audit (if applicable)

**Security Review**
- [ ] Review for common vulnerabilities
- [ ] Input validation audit
- [ ] Dependency security scan (cargo-audit)
- [ ] Document security considerations
- [ ] Set up security disclosure process

**Legal & Licensing**
- [ ] Choose license (MIT or Apache-2.0 recommended)
- [ ] Add LICENSE file
- [ ] Add NOTICE file
- [ ] Review dependency licenses
- [ ] Add copyright headers
- [ ] Create CONTRIBUTING.md

**Release Preparation**
- [ ] Final API review
- [ ] Update all documentation
- [ ] Write 1.0 announcement blog post
- [ ] Prepare demo video (optional)
- [ ] Create comparison benchmarks vs competitors
- [ ] Set up community channels (Discord/Slack)

**Polish**
- [ ] Improve error messages
- [ ] Add helpful CLI messages (if applicable)
- [ ] Improve Python error tracebacks
- [ ] Code cleanup (remove TODOs, dead code)
- [ ] Format all code (rustfmt, black)
- [ ] Lint cleanup (clippy)

**Launch Checklist**
- [ ] All tests passing
- [ ] Benchmarks meet targets
- [ ] Documentation complete
- [ ] Examples working
- [ ] PyPI package published
- [ ] GitHub repo public
- [ ] README polished
- [ ] License in place
- [ ] Security audit complete

**Sprint 14 Review**
- ✅ Production ready
- ✅ 1.0 release candidate
- ✅ Phase 2 complete! 🚀

---

## Phase 3: Ecosystem

**Goal**: Adoption, community building, enterprise features

### Sprint 15+: Community & Growth (Ongoing)

#### Focus Areas

**Community Building**
- [ ] Open source launch (HN, Reddit, Twitter)
- [ ] Create Discord/Slack community
- [ ] Set up GitHub Discussions
- [ ] Write introductory blog posts
- [ ] Create video tutorials
- [ ] Engage with users on issues/PRs

**Enterprise Features** (as needed)
- [ ] Encryption at rest
- [ ] Audit logging
- [ ] Access controls
- [ ] Backup/restore utilities
- [ ] Monitoring/metrics export
- [ ] Multi-tenancy support

**Additional Language Bindings**
- [ ] Node.js bindings (napi-rs)
- [ ] Go bindings (cgo)
- [ ] Other languages as requested

**Advanced Features**
- [ ] Advanced NER for entity extraction
- [ ] LLM-based causal inference
- [ ] Compression for storage efficiency
- [ ] Incremental backups
- [ ] Query caching
- [ ] Hybrid search (BM25 + vector)

**Integrations**
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] Autogen integration
- [ ] Example apps for popular frameworks

**Maintenance**
- [ ] Bug fixes
- [ ] Performance improvements
- [ ] Dependency updates
- [ ] Security patches
- [ ] User support

---

## Risk Management

### Technical Risks

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| Rust learning curve | Start with simple modules, leverage docs | Dev | Ongoing |
| usearch performance issues | Benchmark early (Sprint 2), alternative ready | Dev | Sprint 2 |
| PyO3 complexity | Start with simple bindings, iterate | Dev | Sprint 8 |
| Performance targets missed | Profile continuously, optimize early | Dev | Sprint 10 |
| Scope creep | Strict sprint discipline, defer features | Dev | Ongoing |

### Market Risks

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| Competitor launches first | Focus on differentiation (4D native) | Product | Monitor |
| Low adoption | Build for own use case, dogfood | Product | Phase 3 |
| Feature requests derail | Maintain roadmap discipline | Product | Ongoing |

### Resource Risks

| Risk | Mitigation | Owner | Status |
|------|------------|-------|--------|
| Time constraints | Cut scope, not quality | Dev | Ongoing |
| Burnout | Sustainable pace, celebrate milestones | Dev | Ongoing |
| Knowledge gaps | Learn-as-you-go, leverage community | Dev | Ongoing |

---

## Success Criteria

### Phase 1 (Sprint 8 Exit)

- [ ] All four dimensions functional
- [ ] Python bindings working
- [ ] Can install with maturin develop
- [ ] Basic performance targets met (<10ms search for 100K)
- [ ] Core documentation complete
- [ ] 50+ unit tests passing

### Phase 2 (Sprint 14 Exit)

- [ ] ACID guarantees verified
- [ ] Performance benchmarks published
- [ ] >80% test coverage
- [ ] API stable (1.0 candidate)
- [ ] Published to PyPI
- [ ] Comprehensive documentation
- [ ] Example applications working

### Phase 3 (Long-term)

- [ ] 1K+ GitHub stars
- [ ] 10K+ monthly PyPI downloads
- [ ] 5+ production users
- [ ] External contributions accepted
- [ ] Active community (Discord/Slack)
- [ ] Positive user feedback

---

## Metrics & Tracking

### Sprint Metrics (Track per sprint)

- **Story Points Completed**: Target 20-30 per sprint
- **Velocity**: Rolling 3-sprint average
- **Test Coverage**: Target >80%
- **Bug Count**: Track P0/P1 bugs
- **Documentation %**: APIs documented / total APIs

### Performance Metrics (Track from Sprint 2)

- **Add Latency**: p50, p99 (target <10ms)
- **Search Latency**: p50, p99 (target <10ms for 100K)
- **Memory Usage**: RSS for various dataset sizes
- **Throughput**: Operations per second
- **Storage Efficiency**: MB per 1K memories

### Quality Metrics (Track from Sprint 9)

- **Test Coverage**: % line coverage
- **Crash Rate**: Crashes per 1K operations (target 0)
- **Error Rate**: Errors per 1K operations
- **Recovery Success**: % successful after crash

---

## Appendix: Story Sizing Guide

**1 Point**: Trivial, <2 hours
- Add a simple utility function
- Write documentation
- Update configuration

**3 Points**: Small, 2-4 hours
- Implement a simple struct
- Add a straightforward feature
- Write unit tests for a module

**5 Points**: Medium, 4-8 hours
- Implement a non-trivial feature
- Integration testing
- Moderate refactoring

**8 Points**: Large, 1-2 days
- Implement a major component
- Complex integration
- Significant testing required

**13 Points**: Very Large, 2-3 days
- Implement a subsystem
- Multiple components integration
- Comprehensive testing
- Should consider breaking down

---

## Appendix: Definition of Done

A story is "Done" when:

1. **Code Complete**
   - [ ] Implementation matches acceptance criteria
   - [ ] Code reviewed (self-review at minimum)
   - [ ] No compiler warnings
   - [ ] Clippy (Rust linter) passes

2. **Tested**
   - [ ] Unit tests written and passing
   - [ ] Integration tests written and passing (if applicable)
   - [ ] Manual testing completed
   - [ ] No known bugs

3. **Documented**
   - [ ] Public APIs have rustdoc comments
   - [ ] Python APIs have docstrings
   - [ ] Architecture docs updated (if applicable)
   - [ ] Examples updated (if applicable)

4. **Integrated**
   - [ ] Merged to main branch
   - [ ] CI/CD passing
   - [ ] No regressions introduced

5. **Reviewed**
   - [ ] Demo completed (sprint review)
   - [ ] Acceptance criteria validated
   - [ ] Stakeholder approval (if applicable)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-14 | 1.0 | Initial implementation plan created |

---

**Next Steps:**
1. Review and approve this plan
2. Set up development environment (Sprint 1)
3. Begin implementation
4. Track progress in project management tool (GitHub Projects recommended)
5. Adjust plan as needed based on learnings

**Let's build something amazing! 🚀**
