//! Memory engine - main API entry point
//!
//! The MemoryEngine struct provides the primary interface for interacting
//! with a MnemeFusion database.

use crate::{
    config::Config,
    error::{Error, Result},
    graph::{CausalTraversalResult, GraphManager},
    index::{TemporalIndex, VectorIndex, VectorIndexConfig},
    ingest::IngestionPipeline,
    query::{FusedResult, IntentClassification, QueryPlanner},
    storage::StorageEngine,
    types::{Entity, Memory, MemoryId, Source, Timestamp},
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Main memory engine interface
///
/// This is the primary entry point for all MnemeFusion operations.
/// It coordinates storage, indexing, and retrieval across all dimensions.
pub struct MemoryEngine {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    pipeline: IngestionPipeline,
    query_planner: QueryPlanner,
    config: Config,
}

impl MemoryEngine {
    /// Open or create a memory database
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .mfdb file
    /// * `config` - Configuration options
    ///
    /// # Returns
    ///
    /// A new MemoryEngine instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database file cannot be created or opened
    /// - The file format is invalid
    /// - The configuration is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// let engine = MemoryEngine::open("./brain.mfdb", Config::default()).unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, config: Config) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Open storage
        let storage = Arc::new(StorageEngine::open(path)?);

        // Create vector index configuration from main config
        let vector_config = VectorIndexConfig {
            dimension: config.embedding_dim,
            connectivity: config.hnsw_m,
            expansion_add: config.hnsw_ef_construction,
            expansion_search: config.hnsw_ef_search,
        };

        // Create and load vector index
        let mut vector_index = VectorIndex::new(vector_config, Arc::clone(&storage))?;
        vector_index.load()?;
        let vector_index = Arc::new(RwLock::new(vector_index));

        // Create temporal index
        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));

        // Create and load graph manager
        let mut graph_manager = GraphManager::new();
        crate::graph::persist::load_graph(&mut graph_manager, &storage)?;
        let graph_manager = Arc::new(RwLock::new(graph_manager));

        // Create ingestion pipeline
        let pipeline = IngestionPipeline::new(
            Arc::clone(&storage),
            Arc::clone(&vector_index),
            Arc::clone(&temporal_index),
            Arc::clone(&graph_manager),
            config.entity_extraction_enabled,
        );

        // Create query planner
        let query_planner = QueryPlanner::new(
            Arc::clone(&storage),
            Arc::clone(&vector_index),
            Arc::clone(&temporal_index),
            Arc::clone(&graph_manager),
        );

        Ok(Self {
            storage,
            vector_index,
            temporal_index,
            graph_manager,
            pipeline,
            query_planner,
            config,
        })
    }

    /// Add a new memory to the database
    ///
    /// This will automatically index the memory across all dimensions:
    /// - Semantic (vector similarity)
    /// - Temporal (time-based)
    /// - Entity (if auto-extraction enabled)
    ///
    /// # Arguments
    ///
    /// * `content` - The text content to store
    /// * `embedding` - Vector embedding (must match configured dimension)
    /// * `metadata` - Optional key-value metadata
    /// * `timestamp` - Optional custom timestamp (defaults to now)
    /// * `source` - Optional provenance/source tracking information
    ///
    /// # Returns
    ///
    /// The ID of the created memory
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Embedding dimension doesn't match configuration
    /// - Storage operation fails
    /// - Source serialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # use mnemefusion_core::types::{Source, SourceType};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // Add memory with source tracking
    /// let source = Source::new(SourceType::Conversation)
    ///     .with_id("conv_123")
    ///     .with_confidence(0.95);
    ///
    /// let id = engine.add(
    ///     "Meeting scheduled for next week".to_string(),
    ///     embedding,
    ///     None,
    ///     None,
    ///     Some(source),
    /// ).unwrap();
    /// ```
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<MemoryId> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            let mut mem = Memory::new_with_timestamp(content, embedding, ts);
            if let Some(meta) = metadata {
                mem.metadata = meta;
            }
            mem
        } else {
            let mut mem = Memory::new(content, embedding);
            if let Some(meta) = metadata {
                mem.metadata = meta;
            }
            mem
        };

        // Add source if provided
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Delegate to ingestion pipeline for atomic indexing
        self.pipeline.add(memory)
    }

    /// Retrieve a memory by ID
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to retrieve
    ///
    /// # Returns
    ///
    /// The memory record if found, or None
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None).unwrap();
    /// let memory = engine.get(&id).unwrap();
    /// if let Some(mem) = memory {
    ///     println!("Content: {}", mem.content);
    /// }
    /// ```
    pub fn get(&self, id: &MemoryId) -> Result<Option<Memory>> {
        self.storage.get_memory(id)
    }

    /// Delete a memory by ID
    ///
    /// This will remove the memory from all indexes.
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to delete
    ///
    /// # Returns
    ///
    /// true if the memory was deleted, false if it didn't exist
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None).unwrap();
    /// let deleted = engine.delete(&id).unwrap();
    /// assert!(deleted);
    /// ```
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        // Delegate to ingestion pipeline for atomic cleanup
        self.pipeline.delete(id)
    }

    /// Get the number of memories in the database
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let count = engine.count().unwrap();
    /// println!("Total memories: {}", count);
    /// ```
    pub fn count(&self) -> Result<usize> {
        self.storage.count_memories()
    }

    /// List all memory IDs (for debugging/testing)
    ///
    /// # Warning
    ///
    /// This loads all memory IDs into memory. Use with caution on large databases.
    pub fn list_ids(&self) -> Result<Vec<MemoryId>> {
        self.storage.list_memory_ids()
    }

    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Search for memories by semantic similarity
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector to search for
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (Memory, similarity_score) tuples, sorted by similarity (highest first)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let query_embedding = vec![0.1; 384];
    /// let results = engine.search(&query_embedding, 10).unwrap();
    /// for (memory, score) in results {
    ///     println!("Similarity: {:.3} - {}", score, memory.content);
    /// }
    /// ```
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(Memory, f32)>> {
        // Search vector index
        let vector_results = {
            let index = self.vector_index.read().unwrap();
            index.search(query_embedding, top_k)?
        };

        // Retrieve full memory records using u64 lookup
        let mut results = Vec::with_capacity(vector_results.len());

        for vector_result in vector_results {
            // Look up memory using the u64 key from vector index
            let key = vector_result.id.to_u64();
            if let Some(memory) = self.storage.get_memory_by_u64(key)? {
                results.push((memory, vector_result.similarity));
            }
        }

        Ok(results)
    }

    /// Intelligent multi-dimensional query with intent classification
    ///
    /// This method performs intent-aware retrieval across all dimensions:
    /// - Classifies the query intent (temporal, causal, entity, factual)
    /// - Retrieves results from relevant dimensions
    /// - Fuses results with adaptive weights based on intent
    ///
    /// # Arguments
    ///
    /// * `query_text` - Natural language query text
    /// * `query_embedding` - Vector embedding of the query
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, fused results with full memory records)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("test.mfdb", Config::default()).unwrap();
    /// # let query_embedding = vec![0.1; 384];
    /// let (intent, results) = engine.query(
    ///     "Why was the meeting cancelled?",
    ///     &query_embedding,
    ///     10
    /// ).unwrap();
    ///
    /// println!("Query intent: {:?}", intent.intent);
    /// for result in results {
    ///     println!("Score: {:.3} - {}", result.1.fused_score, result.0.content);
    /// }
    /// ```
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<(IntentClassification, Vec<(Memory, FusedResult)>)> {
        // Execute query using query planner
        let (intent, fused_results) = self.query_planner.query(query_text, query_embedding, limit)?;

        // Retrieve full memory records
        let mut results = Vec::with_capacity(fused_results.len());
        for fused_result in fused_results {
            if let Some(memory) = self.storage.get_memory(&fused_result.id)? {
                results.push((memory, fused_result));
            }
        }

        Ok((intent, results))
    }

    /// Query memories within a time range
    ///
    /// Returns memories whose timestamps fall within the specified range,
    /// sorted by timestamp (newest first).
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the time range (inclusive)
    /// * `end` - End of the time range (inclusive)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (Memory, Timestamp) tuples, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config, Timestamp};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let now = Timestamp::now();
    /// let week_ago = now.subtract_days(7);
    ///
    /// let results = engine.get_range(week_ago, now, 100).unwrap();
    /// for (memory, timestamp) in results {
    ///     println!("{}: {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_range(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        // Query temporal index
        let temporal_results = self.temporal_index.range_query(start, end, limit)?;

        // Retrieve full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                results.push((memory, temp_result.timestamp));
            }
        }

        Ok(results)
    }

    /// Get the N most recent memories
    ///
    /// Returns the most recent memories, sorted by timestamp (newest first).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent memories to retrieve
    ///
    /// # Returns
    ///
    /// A vector of (Memory, Timestamp) tuples, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let recent = engine.get_recent(10).unwrap();
    /// println!("10 most recent memories:");
    /// for (memory, timestamp) in recent {
    ///     println!("  {} - {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_recent(&self, n: usize) -> Result<Vec<(Memory, Timestamp)>> {
        // Query temporal index
        let temporal_results = self.temporal_index.recent(n)?;

        // Retrieve full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                results.push((memory, temp_result.timestamp));
            }
        }

        Ok(results)
    }

    /// Add a causal link between two memories
    ///
    /// Links a cause memory to an effect memory with a confidence score.
    ///
    /// # Arguments
    ///
    /// * `cause` - The MemoryId of the cause
    /// * `effect` - The MemoryId of the effect
    /// * `confidence` - Confidence score (0.0 to 1.0)
    /// * `evidence` - Evidence text explaining the causal relationship
    ///
    /// # Errors
    ///
    /// Returns error if confidence is not in range [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id1 = engine.add("Cause".to_string(), vec![0.1; 384], None, None).unwrap();
    /// # let id2 = engine.add("Effect".to_string(), vec![0.2; 384], None, None).unwrap();
    /// engine.add_causal_link(&id1, &id2, 0.9, "id1 caused id2".to_string()).unwrap();
    /// ```
    pub fn add_causal_link(
        &self,
        cause: &MemoryId,
        effect: &MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<()> {
        let mut graph = self.graph_manager.write().unwrap();
        graph.add_causal_link(cause, effect, confidence, evidence)
    }

    /// Get causes of a memory (backward traversal)
    ///
    /// Finds all memories that causally precede the given memory, up to max_hops.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to find causes for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    ///
    /// CausalTraversalResult with all paths found
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None).unwrap();
    /// let causes = engine.get_causes(&id, 3).unwrap();
    /// for path in causes.paths {
    ///     println!("Found causal path with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_causes(&self, memory_id: &MemoryId, max_hops: usize) -> Result<CausalTraversalResult> {
        let graph = self.graph_manager.read().unwrap();
        graph.get_causes(memory_id, max_hops)
    }

    /// Get effects of a memory (forward traversal)
    ///
    /// Finds all memories that causally follow the given memory, up to max_hops.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to find effects for
    /// * `max_hops` - Maximum traversal depth
    ///
    /// # Returns
    ///
    /// CausalTraversalResult with all paths found
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None).unwrap();
    /// let effects = engine.get_effects(&id, 3).unwrap();
    /// for path in effects.paths {
    ///     println!("Found effect chain with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_effects(&self, memory_id: &MemoryId, max_hops: usize) -> Result<CausalTraversalResult> {
        let graph = self.graph_manager.read().unwrap();
        graph.get_effects(memory_id, max_hops)
    }

    // ========== Entity Operations ==========

    /// Get all memories that mention a specific entity
    ///
    /// # Arguments
    ///
    /// * `entity_name` - The name of the entity to query (case-insensitive)
    ///
    /// # Returns
    ///
    /// A vector of Memory objects that mention this entity
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let memories = engine.get_entity_memories("Project Alpha").unwrap();
    /// for memory in memories {
    ///     println!("{}", memory.content);
    /// }
    /// ```
    pub fn get_entity_memories(&self, entity_name: &str) -> Result<Vec<Memory>> {
        // Find the entity by name
        let entity = self.storage.find_entity_by_name(entity_name)?;

        match entity {
            Some(entity) => {
                // Query entity graph
                let graph = self.graph_manager.read().unwrap();
                let result = graph.get_entity_memories(&entity.id);

                // Retrieve full memory records
                let mut memories = Vec::with_capacity(result.memories.len());
                for memory_id in result.memories {
                    if let Some(memory) = self.storage.get_memory(&memory_id)? {
                        memories.push(memory);
                    }
                }

                Ok(memories)
            }
            None => Ok(Vec::new()), // Entity not found, return empty list
        }
    }

    /// Get all entities mentioned in a specific memory
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory to query
    ///
    /// # Returns
    ///
    /// A vector of Entity objects mentioned in this memory
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("Alice met Bob".to_string(), vec![0.1; 384], None, None).unwrap();
    /// let entities = engine.get_memory_entities(&id).unwrap();
    /// for entity in entities {
    ///     println!("Entity: {}", entity.name);
    /// }
    /// ```
    pub fn get_memory_entities(&self, memory_id: &MemoryId) -> Result<Vec<Entity>> {
        // Query entity graph
        let graph = self.graph_manager.read().unwrap();
        let entity_ids = graph.get_memory_entities(memory_id);

        // Retrieve full entity records
        let mut entities = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            if let Some(entity) = self.storage.get_entity(&entity_id)? {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// List all entities in the database
    ///
    /// # Returns
    ///
    /// A vector of all Entity objects
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let all_entities = engine.list_entities().unwrap();
    /// for entity in all_entities {
    ///     println!("{}: {} mentions", entity.name, entity.mention_count);
    /// }
    /// ```
    pub fn list_entities(&self) -> Result<Vec<Entity>> {
        self.storage.list_entities()
    }

    /// Close the database
    ///
    /// This saves all indexes and ensures all data is flushed to disk.
    /// While not strictly necessary (redb handles persistence automatically),
    /// it's good practice to call this explicitly when you're done.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// // ... use engine ...
    /// engine.close().unwrap();
    /// ```
    pub fn close(self) -> Result<()> {
        // Save vector index
        {
            let index = self.vector_index.read().unwrap();
            index.save()?;
        }

        // Save causal graph
        {
            let graph = self.graph_manager.read().unwrap();
            crate::graph::persist::save_graph(&graph, &self.storage)?;
        }

        // Storage (redb) handles persistence automatically
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_engine_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        assert_eq!(engine.config().embedding_dim, 384);
    }

    #[test]
    fn test_memory_engine_invalid_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let mut config = Config::default();
        config.embedding_dim = 0;

        let result = MemoryEngine::open(&path, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_engine_add_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let content = "Test memory content".to_string();
        let embedding = vec![0.1; 384];

        let id = engine.add(content.clone(), embedding.clone(), None, None, None).unwrap();

        let memory = engine.get(&id).unwrap();
        assert!(memory.is_some());

        let memory = memory.unwrap();
        assert_eq!(memory.content, content);
        assert_eq!(memory.embedding, embedding);
    }

    #[test]
    fn test_memory_engine_invalid_dimension() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let result = engine.add(
            "test".to_string(),
            vec![0.1; 512], // Wrong dimension
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_engine_with_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let id = engine
            .add(
                "test".to_string(),
                vec![0.1; 384],
                Some(metadata),
                None,
                None,
            )
            .unwrap();

        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_memory_engine_with_custom_timestamp() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let ts = Timestamp::from_unix_secs(1609459200.0); // 2021-01-01
        let id = engine
            .add(
                "test".to_string(),
                vec![0.1; 384],
                None,
                Some(ts),
                None,
            )
            .unwrap();

        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.created_at, ts);
    }

    #[test]
    fn test_memory_engine_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None).unwrap();

        let deleted = engine.delete(&id).unwrap();
        assert!(deleted);

        let memory = engine.get(&id).unwrap();
        assert!(memory.is_none());
    }

    #[test]
    fn test_memory_engine_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        assert_eq!(engine.count().unwrap(), 0);

        engine.add("test1".to_string(), vec![0.1; 384], None, None, None).unwrap();
        assert_eq!(engine.count().unwrap(), 1);

        engine.add("test2".to_string(), vec![0.2; 384], None, None, None).unwrap();
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_memory_engine_list_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id1 = engine.add("test1".to_string(), vec![0.1; 384], None, None, None).unwrap();
        let id2 = engine.add("test2".to_string(), vec![0.2; 384], None, None, None).unwrap();

        let ids = engine.list_ids().unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_memory_engine_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let id = {
            let engine = MemoryEngine::open(&path, Config::default()).unwrap();
            let id = engine.add("persistent".to_string(), vec![0.5; 384], None, None, None).unwrap();
            engine.close().unwrap();
            id
        };

        // Reopen and verify
        {
            let engine = MemoryEngine::open(&path, Config::default()).unwrap();
            let memory = engine.get(&id).unwrap();
            assert!(memory.is_some());
            assert_eq!(memory.unwrap().content, "persistent");
        }
    }
}
