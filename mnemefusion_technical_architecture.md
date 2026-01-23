# MnemeFusion: Technical Architecture

## Unified Memory Engine Specification

**Document Version:** 0.1  
**Created:** January 2025  
**Status:** Design Phase

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Storage Layer](#storage-layer)
4. [Index Layer](#index-layer)
5. [Graph Layer](#graph-layer)
6. [Query Planner](#query-planner)
7. [Intent Classification](#intent-classification)
8. [Fusion Algorithm](#fusion-algorithm)
9. [Ingestion Pipeline](#ingestion-pipeline)
10. [Python Bindings](#python-bindings)
11. [File Format](#file-format)
12. [API Reference](#api-reference)
13. [Error Handling](#error-handling)
14. [Performance Considerations](#performance-considerations)
15. [Open Questions](#open-questions)

---

## Overview

### Design Principles

1. **Embedded-first**: No server, no network, library only
2. **Single-file storage**: One portable `.mfdb` file
3. **Four dimensions native**: Semantic, temporal, causal, entity from day one
4. **Intent-aware**: Query understanding drives retrieval strategy
5. **Compose, don't reimplement**: Use proven Rust libraries
6. **Python-friendly**: First-class Python API via PyO3

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python Application                           │
│                                                                      │
│    from mnemefusion import Memory                                   │
│    memory = Memory("brain.mfdb")                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ PyO3 FFI
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MnemeFusion Core (Rust)                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Public API Layer                          │ │
│  │    Memory::new()  ::add()  ::search()  ::close()              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Query Planner                             │ │
│  │    Intent Classification → Weight Selection → Execution Plan   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
│  ┌──────────┬───────────┬───────────┬───────────┬────────────────┐ │
│  │  Vector  │  Temporal │   Causal  │   Entity  │    Payload     │ │
│  │  Index   │   Index   │   Graph   │   Graph   │    Store       │ │
│  │ (usearch)│  (B-tree) │(petgraph) │(petgraph) │   (redb)       │ │
│  └──────────┴───────────┴───────────┴───────────┴────────────────┘ │
│                                  │                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Unified Storage Layer                       │ │
│  │                         (redb)                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                  │                                   │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  brain.mfdb  │
                            │ (single file)│
                            └──────────────┘
```

### Technology Choices

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| Language | Rust | 1.75+ | Memory safety, performance, single binary |
| Vector Index | usearch | 2.x | Battle-tested HNSW, persistence, quantization |
| Storage | redb | 2.x | Pure Rust, ACID, single-file, simple API |
| Graphs | petgraph | 0.6.x | Mature, comprehensive algorithms |
| Serialization | rkyv | 0.7.x | Zero-copy deserialization |
| Python Bindings | PyO3 | 0.20.x | Industry standard, async support |

---

## System Architecture

### Module Structure

```
mnemefusion/
├── Cargo.toml              # Workspace root
├── mnemefusion-core/       # Core Rust library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Public API
│       ├── memory.rs       # Memory struct
│       ├── config.rs       # Configuration
│       ├── error.rs        # Error types
│       ├── storage/
│       │   ├── mod.rs
│       │   ├── engine.rs   # Storage engine (redb)
│       │   └── format.rs   # File format definitions
│       ├── index/
│       │   ├── mod.rs
│       │   ├── vector.rs   # HNSW wrapper (usearch)
│       │   └── temporal.rs # Temporal B-tree index
│       ├── graph/
│       │   ├── mod.rs
│       │   ├── causal.rs   # Causal relationship graph
│       │   ├── entity.rs   # Entity relationship graph
│       │   └── persist.rs  # Graph serialization
│       ├── query/
│       │   ├── mod.rs
│       │   ├── planner.rs  # Query planning
│       │   ├── intent.rs   # Intent classification
│       │   └── fusion.rs   # Result fusion
│       ├── ingest/
│       │   ├── mod.rs
│       │   └── pipeline.rs # Ingestion pipeline
│       └── types/
│           ├── mod.rs
│           ├── memory.rs   # Memory struct
│           ├── entity.rs   # Entity types
│           └── result.rs   # Search results
├── mnemefusion-python/     # Python bindings
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs          # PyO3 bindings
└── tests/
    ├── integration/
    └── benchmarks/
```

### Dependency Graph

```
┌─────────────────┐
│   Python App    │
└────────┬────────┘
         │ PyO3
         ▼
┌─────────────────┐
│mnemefusion-python│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│mnemefusion-core │────►│   usearch   │
└────────┬────────┘     └─────────────┘
         │              ┌─────────────┐
         ├─────────────►│   petgraph  │
         │              └─────────────┘
         │              ┌─────────────┐
         └─────────────►│    redb     │
                        └─────────────┘
```

---

## Storage Layer

### Storage Engine (redb)

MnemeFusion uses redb as the unified storage layer for all persistent data.

**Why redb:**
- Pure Rust (no C dependencies)
- ACID transactions
- Single file storage
- MVCC for concurrent reads
- Simple, focused API

### Table Structure

```rust
use redb::{Database, TableDefinition};

// Memory content and metadata
const MEMORIES: TableDefinition<&[u8], &[u8]> = 
    TableDefinition::new("memories");

// Temporal index (timestamp -> memory_id)
const TEMPORAL_INDEX: TableDefinition<u64, &[u8]> = 
    TableDefinition::new("temporal_index");

// Entity registry (entity_id -> entity_data)
const ENTITIES: TableDefinition<&[u8], &[u8]> = 
    TableDefinition::new("entities");

// Entity name index (name -> entity_id)
const ENTITY_NAMES: TableDefinition<&str, &[u8]> = 
    TableDefinition::new("entity_names");

// Memory-Entity links (composite key -> relationship)
const MEMORY_ENTITIES: TableDefinition<&[u8], &[u8]> = 
    TableDefinition::new("memory_entities");

// Causal edges (cause_id -> Vec<effect_id>)
const CAUSAL_FORWARD: TableDefinition<&[u8], &[u8]> = 
    TableDefinition::new("causal_forward");

// Causal reverse index (effect_id -> Vec<cause_id>)  
const CAUSAL_REVERSE: TableDefinition<&[u8], &[u8]> = 
    TableDefinition::new("causal_reverse");

// Vector index metadata
const VECTOR_META: TableDefinition<&str, &[u8]> = 
    TableDefinition::new("vector_meta");
```

### Storage Engine Interface

```rust
pub struct StorageEngine {
    db: Database,
    path: PathBuf,
}

impl StorageEngine {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let db = Database::create(path.as_ref())?;
        Ok(Self { 
            db,
            path: path.as_ref().to_path_buf(),
        })
    }
    
    pub fn transaction<F, R>(&self, f: F) -> Result<R, Error>
    where
        F: FnOnce(&WriteTransaction) -> Result<R, Error>,
    {
        let txn = self.db.begin_write()?;
        let result = f(&txn)?;
        txn.commit()?;
        Ok(result)
    }
    
    pub fn read<F, R>(&self, f: F) -> Result<R, Error>
    where
        F: FnOnce(&ReadTransaction) -> Result<R, Error>,
    {
        let txn = self.db.begin_read()?;
        f(&txn)
    }
    
    // Memory operations
    pub fn store_memory(&self, memory: &Memory) -> Result<(), Error>;
    pub fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>, Error>;
    pub fn delete_memory(&self, id: &MemoryId) -> Result<bool, Error>;
    
    // Temporal operations
    pub fn temporal_range(
        &self, 
        start: Timestamp, 
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<MemoryId>, Error>;
    
    // Entity operations
    pub fn store_entity(&self, entity: &Entity) -> Result<(), Error>;
    pub fn find_entity(&self, name: &str) -> Result<Option<Entity>, Error>;
    pub fn link_memory_entity(
        &self,
        memory_id: &MemoryId,
        entity_id: &EntityId,
        relationship: &str,
    ) -> Result<(), Error>;
    
    // Causal operations
    pub fn add_causal_link(
        &self,
        cause: &MemoryId,
        effect: &MemoryId,
        confidence: f32,
    ) -> Result<(), Error>;
    
    pub fn get_causes(&self, effect: &MemoryId) -> Result<Vec<CausalLink>, Error>;
    pub fn get_effects(&self, cause: &MemoryId) -> Result<Vec<CausalLink>, Error>;
}
```

---

## Index Layer

### Vector Index (usearch)

Wraps usearch library for HNSW-based vector similarity search.

```rust
use usearch::Index;

pub struct VectorIndex {
    index: Index,
    dimension: usize,
    storage: Arc<StorageEngine>,
}

impl VectorIndex {
    pub fn new(dimension: usize, storage: Arc<StorageEngine>) -> Result<Self, Error> {
        let index = Index::new(&IndexOptions {
            dimensions: dimension,
            metric: MetricKind::Cos,  // Cosine similarity
            quantization: ScalarKind::F32,
            connectivity: 16,         // HNSW M parameter
            expansion_add: 128,       // HNSW ef_construction
            expansion_search: 64,     // HNSW ef_search
        })?;
        
        Ok(Self { index, dimension, storage })
    }
    
    pub fn add(&mut self, id: MemoryId, embedding: &[f32]) -> Result<(), Error> {
        // Convert MemoryId to u64 key for usearch
        let key = id.to_u64();
        self.index.add(key, embedding)?;
        Ok(())
    }
    
    pub fn search(
        &self, 
        query: &[f32], 
        top_k: usize,
    ) -> Result<Vec<VectorResult>, Error> {
        let results = self.index.search(query, top_k)?;
        
        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &distance)| VectorResult {
                id: MemoryId::from_u64(key),
                // Convert distance to similarity (cosine: 1 - distance)
                similarity: 1.0 - distance,
            })
            .collect())
    }
    
    pub fn remove(&mut self, id: &MemoryId) -> Result<(), Error> {
        let key = id.to_u64();
        self.index.remove(key)?;
        Ok(())
    }
    
    pub fn save(&self) -> Result<(), Error> {
        // Serialize index to storage
        let bytes = self.index.save_to_buffer()?;
        self.storage.transaction(|txn| {
            let mut table = txn.open_table(VECTOR_META)?;
            table.insert("index_data", bytes.as_slice())?;
            Ok(())
        })
    }
    
    pub fn load(&mut self) -> Result<(), Error> {
        let bytes = self.storage.read(|txn| {
            let table = txn.open_table(VECTOR_META)?;
            match table.get("index_data")? {
                Some(data) => Ok(Some(data.value().to_vec())),
                None => Ok(None),
            }
        })?;
        
        if let Some(bytes) = bytes {
            self.index.load_from_buffer(&bytes)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VectorResult {
    pub id: MemoryId,
    pub similarity: f32,  // 0.0 to 1.0
}
```

### Temporal Index

B-tree based temporal indexing using redb's native ordering.

```rust
pub struct TemporalIndex {
    storage: Arc<StorageEngine>,
}

impl TemporalIndex {
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self { storage }
    }
    
    pub fn add(&self, id: &MemoryId, timestamp: Timestamp) -> Result<(), Error> {
        self.storage.transaction(|txn| {
            let mut table = txn.open_table(TEMPORAL_INDEX)?;
            // Key is timestamp (allows range queries)
            // Value is memory_id
            table.insert(timestamp.as_micros(), id.as_bytes())?;
            Ok(())
        })
    }
    
    pub fn range_query(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<TemporalResult>, Error> {
        self.storage.read(|txn| {
            let table = txn.open_table(TEMPORAL_INDEX)?;
            let range = table.range(start.as_micros()..=end.as_micros())?;
            
            let mut results = Vec::with_capacity(limit);
            for entry in range.take(limit) {
                let (ts, id_bytes) = entry?;
                results.push(TemporalResult {
                    id: MemoryId::from_bytes(id_bytes.value()),
                    timestamp: Timestamp::from_micros(ts.value()),
                });
            }
            
            // Reverse for most recent first
            results.reverse();
            Ok(results)
        })
    }
    
    pub fn recent(&self, n: usize) -> Result<Vec<TemporalResult>, Error> {
        self.storage.read(|txn| {
            let table = txn.open_table(TEMPORAL_INDEX)?;
            let iter = table.iter()?.rev();  // Reverse iterator
            
            let mut results = Vec::with_capacity(n);
            for entry in iter.take(n) {
                let (ts, id_bytes) = entry?;
                results.push(TemporalResult {
                    id: MemoryId::from_bytes(id_bytes.value()),
                    timestamp: Timestamp::from_micros(ts.value()),
                });
            }
            Ok(results)
        })
    }
}

#[derive(Debug, Clone)]
pub struct TemporalResult {
    pub id: MemoryId,
    pub timestamp: Timestamp,
}
```

---

## Graph Layer

### Graph Manager

Manages both causal and entity graphs using petgraph for in-memory operations with redb persistence.

```rust
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::{dijkstra, has_path_connecting};
use std::collections::HashMap;

pub struct GraphManager {
    // In-memory graphs for fast traversal
    causal_graph: DiGraph<MemoryId, CausalEdge>,
    entity_graph: DiGraph<EntityNode, EntityEdge>,
    
    // Index mappings
    memory_to_causal_node: HashMap<MemoryId, NodeIndex>,
    entity_to_node: HashMap<EntityId, NodeIndex>,
    
    // Persistence
    storage: Arc<StorageEngine>,
    dirty: bool,
}

#[derive(Clone)]
pub struct CausalEdge {
    pub confidence: f32,
    pub evidence: String,
}

#[derive(Clone)]
pub enum EntityNode {
    Entity(Entity),
    Memory(MemoryId),
}

#[derive(Clone)]
pub struct EntityEdge {
    pub relationship: String,  // "mentions", "about", "created_by"
}

impl GraphManager {
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self {
            causal_graph: DiGraph::new(),
            entity_graph: DiGraph::new(),
            memory_to_causal_node: HashMap::new(),
            entity_to_node: HashMap::new(),
            storage,
            dirty: false,
        }
    }
    
    // === Causal Graph Operations ===
    
    pub fn add_causal_link(
        &mut self,
        cause: MemoryId,
        effect: MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<(), Error> {
        // Get or create nodes
        let cause_node = self.get_or_create_causal_node(cause);
        let effect_node = self.get_or_create_causal_node(effect);
        
        // Add edge
        self.causal_graph.add_edge(
            cause_node,
            effect_node,
            CausalEdge { confidence, evidence },
        );
        
        self.dirty = true;
        Ok(())
    }
    
    pub fn get_causes(
        &self,
        effect: &MemoryId,
        max_hops: usize,
    ) -> Vec<CausalTraversalResult> {
        let Some(&node) = self.memory_to_causal_node.get(effect) else {
            return vec![];
        };
        
        self.traverse_backward(node, max_hops)
    }
    
    pub fn get_effects(
        &self,
        cause: &MemoryId,
        max_hops: usize,
    ) -> Vec<CausalTraversalResult> {
        let Some(&node) = self.memory_to_causal_node.get(cause) else {
            return vec![];
        };
        
        self.traverse_forward(node, max_hops)
    }
    
    fn traverse_backward(
        &self,
        start: NodeIndex,
        max_hops: usize,
    ) -> Vec<CausalTraversalResult> {
        use petgraph::visit::Bfs;
        use petgraph::Direction;
        
        let mut results = Vec::new();
        let mut visited = HashMap::new();
        let mut queue = vec![(start, 0, 1.0f32, vec![])];
        
        while let Some((node, hops, cumulative_conf, path)) = queue.pop() {
            if hops > max_hops {
                continue;
            }
            
            // Get incoming edges (causes)
            for edge in self.causal_graph.edges_directed(node, Direction::Incoming) {
                let source = edge.source();
                let edge_data = edge.weight();
                
                if visited.contains_key(&source) {
                    continue;
                }
                
                let memory_id = self.causal_graph[source].clone();
                let new_conf = cumulative_conf * edge_data.confidence;
                let mut new_path = path.clone();
                new_path.push(memory_id.clone());
                
                visited.insert(source, ());
                
                results.push(CausalTraversalResult {
                    memory_id: memory_id.clone(),
                    hop_count: hops + 1,
                    path: new_path.clone(),
                    confidence: new_conf,
                });
                
                queue.push((source, hops + 1, new_conf, new_path));
            }
        }
        
        results
    }
    
    // === Entity Graph Operations ===
    
    pub fn add_entity(&mut self, entity: Entity) -> Result<(), Error> {
        if !self.entity_to_node.contains_key(&entity.id) {
            let node = self.entity_graph.add_node(EntityNode::Entity(entity.clone()));
            self.entity_to_node.insert(entity.id.clone(), node);
        }
        self.dirty = true;
        Ok(())
    }
    
    pub fn link_memory_to_entity(
        &mut self,
        memory_id: MemoryId,
        entity_id: EntityId,
        relationship: String,
    ) -> Result<(), Error> {
        // Get or create memory node
        let memory_node = self.get_or_create_entity_memory_node(memory_id);
        
        // Get entity node
        let Some(&entity_node) = self.entity_to_node.get(&entity_id) else {
            return Err(Error::EntityNotFound(entity_id));
        };
        
        // Add edge from memory to entity
        self.entity_graph.add_edge(
            memory_node,
            entity_node,
            EntityEdge { relationship },
        );
        
        self.dirty = true;
        Ok(())
    }
    
    pub fn get_entity_memories(&self, entity_id: &EntityId) -> Vec<MemoryId> {
        let Some(&node) = self.entity_to_node.get(entity_id) else {
            return vec![];
        };
        
        // Find all memory nodes connected to this entity
        self.entity_graph
            .neighbors_undirected(node)
            .filter_map(|n| {
                match &self.entity_graph[n] {
                    EntityNode::Memory(id) => Some(id.clone()),
                    EntityNode::Entity(_) => None,
                }
            })
            .collect()
    }
    
    // === Persistence ===
    
    pub fn save(&mut self) -> Result<(), Error> {
        if !self.dirty {
            return Ok(());
        }
        
        // Serialize causal graph to storage
        self.save_causal_graph()?;
        
        // Serialize entity graph to storage
        self.save_entity_graph()?;
        
        self.dirty = false;
        Ok(())
    }
    
    pub fn load(&mut self) -> Result<(), Error> {
        self.load_causal_graph()?;
        self.load_entity_graph()?;
        Ok(())
    }
    
    fn save_causal_graph(&self) -> Result<(), Error> {
        // Serialize edges to storage
        self.storage.transaction(|txn| {
            let mut forward = txn.open_table(CAUSAL_FORWARD)?;
            let mut reverse = txn.open_table(CAUSAL_REVERSE)?;
            
            // Clear existing
            // ... (implementation)
            
            // Write edges
            for edge in self.causal_graph.edge_references() {
                let cause_id = &self.causal_graph[edge.source()];
                let effect_id = &self.causal_graph[edge.target()];
                let data = edge.weight();
                
                // Store forward and reverse indexes
                // ... (implementation)
            }
            
            Ok(())
        })
    }
}

#[derive(Debug, Clone)]
pub struct CausalTraversalResult {
    pub memory_id: MemoryId,
    pub hop_count: usize,
    pub path: Vec<MemoryId>,
    pub confidence: f32,
}
```

---

## Query Planner

### Query Planner Structure

```rust
pub struct QueryPlanner {
    intent_classifier: IntentClassifier,
    weight_config: AdaptiveWeightConfig,
}

impl QueryPlanner {
    pub fn plan(&self, query: &str) -> QueryPlan {
        // 1. Classify intent
        let (intent, confidence) = self.intent_classifier.classify(query);
        
        // 2. Get weights for this intent
        let weights = self.weight_config.get_weights(intent);
        
        // 3. Extract signals
        let signals = self.extract_signals(query);
        
        // 4. Build execution plan
        QueryPlan {
            intent,
            intent_confidence: confidence,
            weights,
            signals,
            execution_order: self.determine_execution_order(intent, &signals),
        }
    }
    
    fn extract_signals(&self, query: &str) -> QuerySignals {
        QuerySignals {
            time_range: self.extract_temporal_range(query),
            entity_mentions: self.extract_entities(query),
            keywords: self.extract_keywords(query),
        }
    }
    
    fn determine_execution_order(
        &self,
        intent: QueryIntent,
        signals: &QuerySignals,
    ) -> Vec<IndexType> {
        // Determine which indexes to query and in what order
        match intent {
            QueryIntent::Causal => vec![
                IndexType::Causal,
                IndexType::Semantic,
                IndexType::Entity,
            ],
            QueryIntent::Temporal => vec![
                IndexType::Temporal,
                IndexType::Semantic,
            ],
            QueryIntent::Entity => vec![
                IndexType::Entity,
                IndexType::Semantic,
            ],
            QueryIntent::Factual | QueryIntent::Open => vec![
                IndexType::Semantic,
                IndexType::Temporal,
                IndexType::Entity,
            ],
            QueryIntent::MultiHop => vec![
                IndexType::Semantic,
                IndexType::Entity,
                IndexType::Causal,
            ],
        }
    }
}

#[derive(Debug)]
pub struct QueryPlan {
    pub intent: QueryIntent,
    pub intent_confidence: f32,
    pub weights: IntentWeights,
    pub signals: QuerySignals,
    pub execution_order: Vec<IndexType>,
}

#[derive(Debug)]
pub struct QuerySignals {
    pub time_range: Option<(Timestamp, Timestamp)>,
    pub entity_mentions: Vec<String>,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    Semantic,
    Temporal,
    Causal,
    Entity,
}
```

---

## Intent Classification

### Intent Classifier

```rust
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    Factual,    // Simple fact retrieval
    Temporal,   // "when did X happen"
    Causal,     // "why did X happen"
    Entity,     // "what do we know about X"
    MultiHop,   // Complex multi-fact reasoning
    Open,       // Open-ended, exploratory
}

pub struct IntentClassifier {
    temporal_patterns: Vec<Regex>,
    causal_patterns: Vec<Regex>,
    entity_patterns: Vec<Regex>,
    multi_hop_patterns: Vec<Regex>,
}

impl IntentClassifier {
    pub fn new() -> Self {
        Self {
            temporal_patterns: vec![
                Regex::new(r"(?i)\b(when|what time|what date|how long ago)\b").unwrap(),
                Regex::new(r"(?i)\b(yesterday|today|last week|last month|recently)\b").unwrap(),
                Regex::new(r"(?i)\b(before|after|during|since|until)\b").unwrap(),
                Regex::new(r"(?i)\b(first|last|latest|earliest)\b").unwrap(),
            ],
            causal_patterns: vec![
                Regex::new(r"(?i)\b(why|because|cause|reason|led to|result)\b").unwrap(),
                Regex::new(r"(?i)\b(how come|what made|what caused)\b").unwrap(),
                Regex::new(r"(?i)\b(consequence|effect|impact|due to)\b").unwrap(),
            ],
            entity_patterns: vec![
                Regex::new(r"(?i)\b(who|whom|whose)\b").unwrap(),
                Regex::new(r"(?i)\bwhat (do|does|did) .+ (think|say|believe|want)\b").unwrap(),
                Regex::new(r"(?i)\b(tell me about|what about|regarding)\b").unwrap(),
            ],
            multi_hop_patterns: vec![
                Regex::new(r"(?i)\b(both|all|compare|difference|similar)\b").unwrap(),
                Regex::new(r"(?i)\b(how many|how much|count|total)\b").unwrap(),
                Regex::new(r"(?i)\b(relationship between|connection)\b").unwrap(),
            ],
        }
    }
    
    pub fn classify(&self, query: &str) -> (QueryIntent, f32) {
        let scores = [
            (QueryIntent::Temporal, self.score(&self.temporal_patterns, query)),
            (QueryIntent::Causal, self.score(&self.causal_patterns, query)),
            (QueryIntent::Entity, self.score(&self.entity_patterns, query)),
            (QueryIntent::MultiHop, self.score(&self.multi_hop_patterns, query)),
        ];
        
        let (intent, score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        
        if *score < 0.3 {
            // Low confidence - default to Factual or Open
            if query.contains('?') && query.split_whitespace().count() < 10 {
                (QueryIntent::Factual, 0.5)
            } else {
                (QueryIntent::Open, 0.4)
            }
        } else {
            (*intent, *score)
        }
    }
    
    fn score(&self, patterns: &[Regex], query: &str) -> f32 {
        let matches = patterns.iter().filter(|p| p.is_match(query)).count();
        (matches as f32 * 0.3).min(1.0)
    }
    
    pub fn extract_temporal_range(&self, query: &str) -> Option<(Timestamp, Timestamp)> {
        let query_lower = query.to_lowercase();
        let now = Timestamp::now();
        
        if query_lower.contains("yesterday") {
            let start = now.subtract_days(1).start_of_day();
            let end = now.subtract_days(1).end_of_day();
            return Some((start, end));
        }
        
        if query_lower.contains("last week") {
            let start = now.subtract_days(7);
            return Some((start, now));
        }
        
        if query_lower.contains("last month") {
            let start = now.subtract_days(30);
            return Some((start, now));
        }
        
        if query_lower.contains("today") {
            let start = now.start_of_day();
            let end = now.end_of_day();
            return Some((start, end));
        }
        
        None
    }
    
    pub fn extract_entity_mentions(&self, query: &str) -> Vec<String> {
        // Simple: extract capitalized words that aren't common
        let stop_words = ["What", "When", "Where", "Who", "Why", "How", "The", "A", "An"];
        
        query
            .split_whitespace()
            .filter(|word| {
                word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !stop_words.contains(word)
            })
            .map(|s| s.to_string())
            .collect()
    }
}
```

---

## Fusion Algorithm

### Adaptive Fusion

```rust
#[derive(Debug, Clone, Copy)]
pub struct IntentWeights {
    pub semantic: f32,
    pub temporal: f32,
    pub causal: f32,
    pub entity: f32,
}

impl IntentWeights {
    pub fn normalize(&mut self) {
        let total = self.semantic + self.temporal + self.causal + self.entity;
        if total > 0.0 {
            self.semantic /= total;
            self.temporal /= total;
            self.causal /= total;
            self.entity /= total;
        }
    }
}

pub struct AdaptiveWeightConfig {
    factual: IntentWeights,
    temporal: IntentWeights,
    causal: IntentWeights,
    entity: IntentWeights,
    multi_hop: IntentWeights,
    open: IntentWeights,
}

impl Default for AdaptiveWeightConfig {
    fn default() -> Self {
        Self {
            factual: IntentWeights {
                semantic: 0.6, temporal: 0.1, causal: 0.1, entity: 0.2,
            },
            temporal: IntentWeights {
                semantic: 0.2, temporal: 0.6, causal: 0.1, entity: 0.1,
            },
            causal: IntentWeights {
                semantic: 0.2, temporal: 0.1, causal: 0.6, entity: 0.1,
            },
            entity: IntentWeights {
                semantic: 0.2, temporal: 0.1, causal: 0.1, entity: 0.6,
            },
            multi_hop: IntentWeights {
                semantic: 0.35, temporal: 0.15, causal: 0.2, entity: 0.3,
            },
            open: IntentWeights {
                semantic: 0.3, temporal: 0.2, causal: 0.2, entity: 0.3,
            },
        }
    }
}

pub struct FusionEngine {
    temporal_decay: TemporalDecayConfig,
}

impl FusionEngine {
    pub fn fuse(
        &self,
        weights: IntentWeights,
        semantic_results: Vec<VectorResult>,
        temporal_results: Vec<TemporalResult>,
        causal_results: Vec<CausalTraversalResult>,
        entity_memories: Vec<MemoryId>,
    ) -> Vec<FusedResult> {
        // Collect all candidate memory IDs
        let mut candidates: HashMap<MemoryId, CandidateScores> = HashMap::new();
        
        // Process semantic results
        for result in semantic_results {
            let scores = candidates.entry(result.id.clone()).or_default();
            scores.semantic = result.similarity;
            scores.sources.push("semantic");
        }
        
        // Process temporal results
        let now = Timestamp::now();
        for result in temporal_results {
            let scores = candidates.entry(result.id.clone()).or_default();
            let age_seconds = now.seconds_since(&result.timestamp);
            scores.temporal = self.temporal_decay.calculate(age_seconds);
            scores.sources.push("temporal");
        }
        
        // Process causal results
        for result in causal_results {
            let scores = candidates.entry(result.memory_id.clone()).or_default();
            scores.causal = self.causal_score(result.hop_count, result.confidence);
            scores.sources.push("causal");
        }
        
        // Process entity results
        for memory_id in entity_memories {
            let scores = candidates.entry(memory_id).or_default();
            scores.entity = 0.8;  // Fixed score for entity match
            if !scores.sources.contains(&"entity") {
                scores.sources.push("entity");
            }
        }
        
        // Calculate fused scores
        let mut results: Vec<FusedResult> = candidates
            .into_iter()
            .map(|(id, scores)| {
                let fused = self.calculate_fused_score(&scores, &weights);
                FusedResult {
                    memory_id: id,
                    semantic_score: scores.semantic,
                    temporal_score: scores.temporal,
                    causal_score: scores.causal,
                    entity_score: scores.entity,
                    fused_score: fused,
                    sources: scores.sources,
                }
            })
            .collect();
        
        // Sort by fused score
        results.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap());
        
        results
    }
    
    fn calculate_fused_score(
        &self,
        scores: &CandidateScores,
        weights: &IntentWeights,
    ) -> f32 {
        // Only consider active dimensions (non-zero scores)
        let mut active_weight_sum = 0.0;
        let mut weighted_sum = 0.0;
        
        if scores.semantic > 0.0 {
            weighted_sum += weights.semantic * scores.semantic;
            active_weight_sum += weights.semantic;
        }
        
        if scores.temporal > 0.0 {
            weighted_sum += weights.temporal * scores.temporal;
            active_weight_sum += weights.temporal;
        }
        
        if scores.causal > 0.0 {
            weighted_sum += weights.causal * scores.causal;
            active_weight_sum += weights.causal;
        }
        
        if scores.entity > 0.0 {
            weighted_sum += weights.entity * scores.entity;
            active_weight_sum += weights.entity;
        }
        
        if active_weight_sum > 0.0 {
            weighted_sum / active_weight_sum
        } else {
            0.0
        }
    }
    
    fn causal_score(&self, hop_count: usize, confidence: f32) -> f32 {
        let hop_factor = 1.0 - (hop_count as f32 / 5.0).min(1.0);
        hop_factor * confidence
    }
}

#[derive(Default)]
struct CandidateScores {
    semantic: f32,
    temporal: f32,
    causal: f32,
    entity: f32,
    sources: Vec<&'static str>,
}

#[derive(Debug)]
pub struct FusedResult {
    pub memory_id: MemoryId,
    pub semantic_score: f32,
    pub temporal_score: f32,
    pub causal_score: f32,
    pub entity_score: f32,
    pub fused_score: f32,
    pub sources: Vec<&'static str>,
}

pub struct TemporalDecayConfig {
    pub half_life_hours: f32,
    pub min_score: f32,
}

impl TemporalDecayConfig {
    pub fn calculate(&self, age_seconds: f64) -> f32 {
        let half_life_seconds = self.half_life_hours * 3600.0;
        let decay = (-0.693 * age_seconds as f32 / half_life_seconds).exp();
        decay.max(self.min_score)
    }
}
```

---

## Ingestion Pipeline

### Ingestion Flow

```rust
pub struct IngestionPipeline {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    entity_extractor: Box<dyn EntityExtractor>,
}

impl IngestionPipeline {
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
        timestamp: Option<Timestamp>,
    ) -> Result<MemoryId, Error> {
        let id = MemoryId::new();
        let timestamp = timestamp.unwrap_or_else(Timestamp::now);
        
        // 1. Create memory record
        let memory = Memory {
            id: id.clone(),
            content: content.clone(),
            embedding: embedding.clone(),
            created_at: timestamp,
            metadata,
        };
        
        // 2. Store memory (atomic)
        self.storage.store_memory(&memory)?;
        
        // 3. Add to vector index
        {
            let mut index = self.vector_index.write().unwrap();
            index.add(id.clone(), &embedding)?;
        }
        
        // 4. Add to temporal index
        self.temporal_index.add(&id, timestamp)?;
        
        // 5. Extract and link entities
        let entities = self.entity_extractor.extract(&content)?;
        {
            let mut graph = self.graph_manager.write().unwrap();
            for entity in entities {
                graph.add_entity(entity.clone())?;
                graph.link_memory_to_entity(
                    id.clone(),
                    entity.id,
                    "mentions".to_string(),
                )?;
            }
        }
        
        Ok(id)
    }
    
    pub fn add_causal_link(
        &self,
        cause: MemoryId,
        effect: MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<(), Error> {
        let mut graph = self.graph_manager.write().unwrap();
        graph.add_causal_link(cause, effect, confidence, evidence)
    }
}
```

---

## Python Bindings

### PyO3 Module

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass]
pub struct Memory {
    inner: Arc<MemoryEngine>,
}

#[pymethods]
impl Memory {
    #[new]
    #[pyo3(signature = (path, config=None))]
    pub fn new(path: &str, config: Option<PyConfig>) -> PyResult<Self> {
        let config = config.map(|c| c.into()).unwrap_or_default();
        let inner = MemoryEngine::open(path, config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(inner) })
    }
    
    pub fn add(
        &self,
        py: Python<'_>,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<f64>,
    ) -> PyResult<String> {
        let timestamp = timestamp.map(|t| Timestamp::from_unix_secs(t));
        let metadata = metadata.unwrap_or_default();
        
        let id = self.inner
            .add(content, embedding, metadata, timestamp)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(id.to_string())
    }
    
    #[pyo3(signature = (query, embedding, top_k=10))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: String,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        let results = self.inner
            .search(&query, &embedding, top_k)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(results.into_iter().map(|r| r.into()).collect())
    }
    
    pub fn get(&self, id: String) -> PyResult<Option<PyMemory>> {
        let memory_id = MemoryId::parse(&id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let memory = self.inner
            .get(&memory_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(memory.map(|m| m.into()))
    }
    
    pub fn get_causes(&self, id: String, max_hops: Option<usize>) -> PyResult<Vec<String>> {
        let memory_id = MemoryId::parse(&id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let causes = self.inner
            .get_causes(&memory_id, max_hops.unwrap_or(2))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(causes.into_iter().map(|c| c.memory_id.to_string()).collect())
    }
    
    pub fn add_causal_link(
        &self,
        cause_id: String,
        effect_id: String,
        confidence: f32,
        evidence: String,
    ) -> PyResult<()> {
        let cause = MemoryId::parse(&cause_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let effect = MemoryId::parse(&effect_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.inner
            .add_causal_link(cause, effect, confidence, evidence)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    pub fn close(&self) -> PyResult<()> {
        self.inner
            .close()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub semantic_score: f32,
    #[pyo3(get)]
    pub temporal_score: f32,
    #[pyo3(get)]
    pub causal_score: f32,
    #[pyo3(get)]
    pub entity_score: f32,
    #[pyo3(get)]
    pub fused_score: f32,
    #[pyo3(get)]
    pub sources: Vec<String>,
}

#[pyclass]
#[derive(Clone)]
pub struct PyMemory {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub created_at: f64,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pyclass]
pub struct PyConfig {
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub temporal_decay_hours: f32,
    #[pyo3(get, set)]
    pub causal_max_hops: usize,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (embedding_dim=384, temporal_decay_hours=168.0, causal_max_hops=3))]
    pub fn new(
        embedding_dim: usize,
        temporal_decay_hours: f32,
        causal_max_hops: usize,
    ) -> Self {
        Self {
            embedding_dim,
            temporal_decay_hours,
            causal_max_hops,
        }
    }
}

#[pymodule]
fn mnemefusion(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyMemory>()?;
    Ok(())
}
```

---

## File Format

### Single File Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    brain.mfdb                                │
├─────────────────────────────────────────────────────────────┤
│  Header (64 bytes)                                          │
│  ├─ Magic: "MFDB" (4 bytes)                                 │
│  ├─ Version: u32 (4 bytes)                                  │
│  ├─ Flags: u64 (8 bytes)                                    │
│  ├─ Created: u64 timestamp (8 bytes)                        │
│  ├─ Modified: u64 timestamp (8 bytes)                       │
│  └─ Reserved (32 bytes)                                     │
├─────────────────────────────────────────────────────────────┤
│  redb Database                                              │
│  ├─ memories table                                          │
│  ├─ temporal_index table                                    │
│  ├─ entities table                                          │
│  ├─ entity_names table                                      │
│  ├─ memory_entities table                                   │
│  ├─ causal_forward table                                    │
│  ├─ causal_reverse table                                    │
│  └─ vector_meta table (serialized HNSW index)              │
└─────────────────────────────────────────────────────────────┘
```

### Format Implementation

```rust
const MAGIC: &[u8; 4] = b"MFDB";
const VERSION: u32 = 1;

#[derive(Debug)]
pub struct FileHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u64,
    pub created_at: u64,
    pub modified_at: u64,
    pub reserved: [u8; 32],
}

impl FileHeader {
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            magic: *MAGIC,
            version: VERSION,
            flags: 0,
            created_at: now,
            modified_at: now,
            reserved: [0; 32],
        }
    }
    
    pub fn validate(&self) -> Result<(), Error> {
        if &self.magic != MAGIC {
            return Err(Error::InvalidFormat("Bad magic number"));
        }
        if self.version > VERSION {
            return Err(Error::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}
```

---

## API Reference

### Python API

```python
from mnemefusion import Memory, Config

# Create/open a memory database
memory = Memory(
    path: str,                    # Path to .mfdb file
    config: Optional[Config]      # Optional configuration
)

# Configuration
config = Config(
    embedding_dim: int = 384,     # Dimension of embeddings
    temporal_decay_hours: float = 168.0,  # Half-life for temporal decay
    causal_max_hops: int = 3,     # Max hops for causal traversal
)

# Add a memory
memory_id: str = memory.add(
    content: str,                 # Memory content
    embedding: List[float],       # Embedding vector
    metadata: Optional[Dict[str, str]] = None,  # Key-value metadata
    timestamp: Optional[float] = None,  # Unix timestamp
)

# Search memories
results: List[SearchResult] = memory.search(
    query: str,                   # Query text (for intent classification)
    embedding: List[float],       # Query embedding vector
    top_k: int = 10,              # Number of results
)

# SearchResult fields:
#   id: str
#   content: str
#   semantic_score: float
#   temporal_score: float
#   causal_score: float
#   entity_score: float
#   fused_score: float
#   sources: List[str]

# Get a specific memory
memory_item: Optional[MemoryItem] = memory.get(id: str)

# MemoryItem fields:
#   id: str
#   content: str
#   created_at: float
#   metadata: Dict[str, str]

# Add causal relationship
memory.add_causal_link(
    cause_id: str,
    effect_id: str,
    confidence: float,            # 0.0 to 1.0
    evidence: str,                # Why we believe this
)

# Get causal chain
causes: List[str] = memory.get_causes(id: str, max_hops: int = 2)
effects: List[str] = memory.get_effects(id: str, max_hops: int = 2)

# Get entity memories
memories: List[str] = memory.get_entity_memories(entity_name: str)

# Close database
memory.close()
```

### Example Usage

```python
from mnemefusion import Memory
from sentence_transformers import SentenceTransformer

# Load embedding model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Open memory database
memory = Memory("./my_memory.mfdb")

# Add some memories
texts = [
    "Project deadline was moved to March 15th",
    "Budget was cut by 20% due to Q4 results",
    "Team morale has been low since the layoffs",
    "The layoffs happened on January 10th",
]

memory_ids = []
for text in texts:
    embedding = encoder.encode(text).tolist()
    mid = memory.add(text, embedding)
    memory_ids.append(mid)
    print(f"Added: {mid[:8]}... - {text}")

# Add causal relationships
# layoffs → low morale
memory.add_causal_link(
    cause_id=memory_ids[3],     # layoffs
    effect_id=memory_ids[2],   # low morale
    confidence=0.9,
    evidence="Temporal proximity and semantic relationship"
)

# budget cut → layoffs
memory.add_causal_link(
    cause_id=memory_ids[1],    # budget cut
    effect_id=memory_ids[3],   # layoffs
    confidence=0.8,
    evidence="Budget cuts often lead to layoffs"
)

# Search with different intents
queries = [
    "Why is team morale low?",           # Causal intent
    "What happened in January?",          # Temporal intent
    "Tell me about the project",          # Entity intent
    "What do we know about deadlines?",   # Factual intent
]

for query in queries:
    print(f"\nQuery: {query}")
    embedding = encoder.encode(query).tolist()
    results = memory.search(query, embedding, top_k=3)
    
    for r in results:
        print(f"  [{r.fused_score:.2f}] {r.content[:50]}...")
        print(f"    Sources: {r.sources}")

memory.close()
```

---

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Storage error: {0}")]
    Storage(#[from] redb::Error),
    
    #[error("Vector index error: {0}")]
    VectorIndex(String),
    
    #[error("Invalid file format: {0}")]
    InvalidFormat(&'static str),
    
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),
    
    #[error("Entity not found: {0}")]
    EntityNotFound(EntityId),
    
    #[error("Invalid embedding dimension: expected {expected}, got {got}")]
    InvalidEmbeddingDimension { expected: usize, got: usize },
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

---

## Performance Considerations

### Expected Performance

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Add memory | < 10ms | Dominated by index updates |
| Search (100K memories) | < 10ms | HNSW + fusion |
| Search (1M memories) | < 50ms | May need quantization |
| Get by ID | < 1ms | Direct lookup |
| Causal traversal (2 hops) | < 5ms | Graph traversal |

### Memory Usage

| Component | Memory per 100K memories |
|-----------|-------------------------|
| Vector index (384-dim) | ~150MB |
| Temporal index | ~10MB |
| Graph structures | ~20MB |
| Payload cache | Configurable |

### Optimization Opportunities

1. **Vector quantization**: Use f16 or i8 to reduce memory 50-75%
2. **Index caching**: Keep hot portions of indexes in memory
3. **Batch operations**: Amortize persistence costs
4. **Lazy graph loading**: Load graph edges on demand

---

## Scaling & Limits

### Design Philosophy

MnemeFusion follows the **SQLite scaling model**: optimize for single-user/single-context performance, and scale horizontally by deploying many instances. This is not a limitation—it's a deliberate architectural choice that enables:

- Predictable performance characteristics
- Simple deployment (no coordination overhead)
- Data isolation by design
- Edge/offline capability

**Target deployment pattern:**
```
NOT: One massive database for all users
YES: One database per user/context, managed by application
```

### Component Limits

#### Storage Layer (redb)

| Aspect | Limit | Notes |
|--------|-------|-------|
| Max file size | ~64 TB | Filesystem dependent |
| Max records | Billions | Key-value store scales well |
| Single record size | ~4 GB | redb limit per value |
| Concurrent readers | Unlimited | MVCC |
| Concurrent writers | 1 | Single-writer model |

**Storage is not the bottleneck.** redb handles large datasets efficiently.

#### Vector Index (usearch/HNSW)

| Aspect | Limit | Notes |
|--------|-------|-------|
| Max vectors | ~10-100 million | Memory bound |
| Memory per vector (384-dim) | ~1.5 KB | With HNSW overhead |
| Memory per vector (768-dim) | ~3 KB | With HNSW overhead |
| Memory per vector (1536-dim) | ~6 KB | With HNSW overhead |

**Memory requirements by scale (768-dim embeddings):**

| Memories | RAM for Vectors | Notes |
|----------|-----------------|-------|
| 10K | ~30 MB | Trivial |
| 100K | ~300 MB | Comfortable |
| 500K | ~1.5 GB | Workable |
| 1M | ~3 GB | Needs dedicated memory |
| 10M | ~30 GB | Requires optimization |

**Vector index is memory-bound. This is the primary constraint for most deployments.**

#### Graph Layer (petgraph)

| Aspect | Limit | Notes |
|--------|-------|-------|
| Storage | In-memory | Current design loads entire graph into RAM |
| Node overhead | ~64 bytes | Per node |
| Edge overhead | ~48 bytes | Per edge |

**Memory requirements by scale:**

| Nodes | Edges (5× nodes) | RAM | Notes |
|-------|------------------|-----|-------|
| 10K | 50K | ~5 MB | Trivial |
| 100K | 500K | ~50 MB | Comfortable |
| 500K | 2.5M | ~250 MB | Workable |
| 1M | 5M | ~500 MB | Getting heavy |
| 10M | 50M | ~5 GB | Problem |

**Graph is the secondary constraint.** For most use cases, graph is smaller than vectors.

### Practical Limits

Combining all components, here are practical limits for typical deployments:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRACTICAL LIMITS                              │
│                                                                  │
│   Sweet spot:       10K - 100K memories per file                │
│   Comfortable:      100K - 500K memories per file               │
│   Workable:         500K - 1M memories per file                 │
│   Needs care:       1M+ memories per file                       │
│                                                                  │
│   Typical file sizes:                                           │
│   • 10K memories:   ~50-100 MB                                  │
│   • 100K memories:  ~500 MB - 1 GB                              │
│   • 1M memories:    ~5-10 GB                                    │
│                                                                  │
│   RAM requirements (768-dim):                                   │
│   • 10K memories:   ~50 MB                                      │
│   • 100K memories:  ~400 MB                                     │
│   • 1M memories:    ~4 GB                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Use Case Fit

| Use Case | Memories/User | Users/File | Total | Verdict |
|----------|---------------|------------|-------|--------|
| Personal AI assistant | 1K-10K | 1 | 10K | ✅ Perfect fit |
| Session memory | 100-1K | 1 | 1K | ✅ Perfect fit |
| Project knowledge base | 10K-50K | 1 | 50K | ✅ Comfortable |
| Enterprise per-user | 10K-100K | 1 | 100K | ✅ Comfortable |
| Team shared memory | 50K-200K | 1 | 200K | ✅ Workable |
| Large knowledge base | 500K-1M | 1 | 1M | ⚠️ Needs optimization |
| Global corpus | 10M+ | 1 | 10M+ | ❌ Wrong tool |
| Multi-user single file | 10K × 1000 | 1000 | 10M | ❌ Use file-per-user |

### Recommended Deployment Patterns

#### Pattern 1: File-Per-User (Recommended)

```python
class UserMemoryManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self._cache: Dict[str, Memory] = {}
    
    def get_memory(self, user_id: str) -> Memory:
        if user_id not in self._cache:
            path = self.base_path / f"{user_id}.mfdb"
            self._cache[user_id] = Memory(str(path))
        return self._cache[user_id]
    
    def close_user(self, user_id: str):
        if user_id in self._cache:
            self._cache[user_id].close()
            del self._cache[user_id]
```

**Benefits:**
- Natural isolation
- Predictable performance
- Easy backup/restore per user
- User deletion = file deletion
- No coordination needed

#### Pattern 2: File-Per-Context

```python
# Different contexts for same user
work_memory = Memory(f"./data/{user_id}_work.mfdb")
personal_memory = Memory(f"./data/{user_id}_personal.mfdb")
project_memory = Memory(f"./data/{user_id}_project_{project_id}.mfdb")
```

**Use when:**
- User has distinct contexts that shouldn't mix
- Different retention policies per context
- Different sharing requirements

#### Pattern 3: Sharded by Time

```python
# For append-heavy workloads with time-based access patterns
def get_memory_for_date(user_id: str, date: datetime) -> Memory:
    year_month = date.strftime("%Y-%m")
    path = f"./data/{user_id}/{year_month}.mfdb"
    return Memory(path)
```

**Use when:**
- Queries are primarily time-bounded
- Old data can be archived/deleted
- Very high write volume

### Scaling Strategies

#### Strategy 1: Stay Within Limits (Recommended for v1)

Design application to work within comfortable limits:

```python
MAX_MEMORIES_PER_FILE = 100_000

def add_memory_with_limit(memory: Memory, content: str, embedding: List[float]):
    # Check current count (implement via metadata or counter)
    if memory.count() >= MAX_MEMORIES_PER_FILE:
        # Option A: Reject
        raise MemoryLimitExceeded()
        
        # Option B: Archive old memories
        archive_oldest_memories(memory, count=10_000)
        
        # Option C: Start new file
        return rotate_memory_file(memory)
    
    return memory.add(content, embedding)
```

#### Strategy 2: Memory-Mapped Vectors (v1.1+)

Enable usearch memory-mapping to remove vector RAM constraint:

```rust
// Instead of loading into RAM
let index = Index::new(...)?;
index.load("vectors.usearch")?;  // Loads into RAM

// Use memory-mapping
let index = Index::new(...)?;
index.view("vectors.usearch")?;  // Memory-mapped, OS manages paging
```

**Impact:** Vector index no longer RAM-bound. Can handle 10M+ vectors with minimal RAM.

**Tradeoff:** Slightly higher latency for cold queries.

#### Strategy 3: Lazy Graph Loading (v1.2+)

Don't load entire graph into memory. Load on-demand:

```
Current:
  Open file → Load all graph nodes/edges into petgraph → Ready

Improved:
  Open file → Graph stays on disk → Load subgraphs on query → Cache hot nodes
```

**Impact:** Graph no longer RAM-bound.

**Effort:** Significant redesign of graph layer.

#### Strategy 4: Tiered Storage (v2.0+)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIERED STORAGE                                │
│                                                                  │
│   Hot tier (RAM):      Last 1K memories                         │
│   • Full vectors in memory                                      │
│   • Full graph in memory                                        │
│   • <10ms queries                                               │
│                                                                  │
│   Warm tier (Memory-mapped):  Next 100K memories                │
│   • Memory-mapped vectors                                       │
│   • On-demand graph loading                                     │
│   • <50ms queries                                               │
│                                                                  │
│   Cold tier (Disk):    Archived memories                        │
│   • Compressed storage                                          │
│   • Explicit retrieval required                                 │
│   • <500ms queries                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Impact:** Handle millions of memories per file.

**Effort:** Major architectural change.

### Configuration for Scale

```python
from mnemefusion import Memory, Config

# For small deployments (< 50K memories)
small_config = Config(
    embedding_dim=384,          # Smaller embeddings
    hnsw_ef_construction=128,   # Standard
    hnsw_ef_search=64,          # Standard
    cache_size_mb=100,          # Minimal cache
)

# For medium deployments (50K - 500K memories)
medium_config = Config(
    embedding_dim=768,          # Standard embeddings
    hnsw_ef_construction=200,   # Better index quality
    hnsw_ef_search=100,         # Better search quality
    cache_size_mb=500,          # Larger cache
    use_mmap_vectors=True,      # Memory-map vectors (when available)
)

# For large deployments (500K+ memories)
large_config = Config(
    embedding_dim=768,
    hnsw_ef_construction=200,
    hnsw_ef_search=100,
    cache_size_mb=1000,
    use_mmap_vectors=True,
    lazy_graph_loading=True,    # Load graph on-demand (when available)
    vector_quantization="f16",  # Reduce memory 50%
)
```

### What MnemeFusion is NOT For

Be explicit about anti-patterns:

| Anti-Pattern | Why It Fails | Alternative |
|--------------|--------------|-------------|
| Global knowledge base with 100M+ documents | Exceeds single-file limits | Use distributed vector DB |
| Real-time multi-writer | Single-writer model | Use server-based solution |
| Multi-tenant single file | No isolation, coordination overhead | File-per-tenant |
| Petabyte-scale storage | Not designed for this | Use data warehouse |

### Monitoring & Alerts

Applications should monitor and alert on:

```python
class MemoryHealthCheck:
    def __init__(self, memory: Memory, thresholds: dict):
        self.memory = memory
        self.thresholds = thresholds
    
    def check(self) -> List[Alert]:
        alerts = []
        stats = self.memory.stats()  # Future API
        
        if stats.memory_count > self.thresholds['max_memories']:
            alerts.append(Alert(
                level='warning',
                message=f"Memory count {stats.memory_count} exceeds threshold"
            ))
        
        if stats.file_size_mb > self.thresholds['max_file_size_mb']:
            alerts.append(Alert(
                level='warning', 
                message=f"File size {stats.file_size_mb}MB exceeds threshold"
            ))
        
        if stats.ram_usage_mb > self.thresholds['max_ram_mb']:
            alerts.append(Alert(
                level='critical',
                message=f"RAM usage {stats.ram_usage_mb}MB exceeds threshold"
            ))
        
        return alerts
```

### Summary

| Question | Answer |
|----------|--------|
| Max memories per file? | Comfortable: 100K. Workable: 1M. |
| Max file size? | ~64TB (not the constraint) |
| Main constraint? | RAM for vectors and graphs |
| How to scale? | File-per-user pattern |
| When to use something else? | 10M+ memories in single queryable corpus |

MnemeFusion is designed for **many small-to-medium memory stores**, not one massive global store. This is the SQLite philosophy applied to AI memory.

---

## Open Questions

### To Resolve in Implementation

| Question | Options | Decision Point |
|----------|---------|----------------|
| Vector library | usearch vs hora | Benchmark in Month 1 |
| Graph persistence | Custom vs cozo | Prototype in Month 3 |
| Embedding handling | Built-in vs external | API design in Month 4 |
| Entity extraction | Built-in NER vs external | Phase 2 |
| Causal inference | Built-in LLM vs external | Phase 2 |

### Deferred to Later

| Feature | Rationale for Deferral |
|---------|----------------------|
| Distributed mode | Focus on embedded first |
| Encryption at rest | Phase 2+ |
| Compression | Optimize after correctness |
| Incremental backup | Phase 2 |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-01-13 | Initial architecture design |
