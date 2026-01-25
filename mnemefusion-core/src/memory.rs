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
    types::{
        AddResult, BatchResult, Entity, Memory, MemoryId, MemoryInput, MetadataFilter, Source,
        Timestamp, UpsertResult,
    },
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
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
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

        // Set namespace if provided (defaults to empty string)
        memory.set_namespace(namespace.unwrap_or(""));

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
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
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
    /// * `namespace` - Optional namespace. If provided, verifies the memory is in this namespace before deleting
    ///
    /// # Returns
    ///
    /// true if the memory was deleted, false if it didn't exist
    ///
    /// # Errors
    ///
    /// Returns `Error::NamespaceMismatch` if namespace is provided and doesn't match
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let deleted = engine.delete(&id, None).unwrap();
    /// assert!(deleted);
    /// ```
    pub fn delete(&self, id: &MemoryId, namespace: Option<&str>) -> Result<bool> {
        // If namespace is provided, verify it matches before deleting
        if let Some(expected_ns) = namespace {
            if let Some(memory) = self.storage.get_memory(id)? {
                let found_ns = memory.get_namespace();
                if found_ns != expected_ns {
                    return Err(Error::NamespaceMismatch {
                        expected: expected_ns.to_string(),
                        found: found_ns,
                    });
                }
            } else {
                // Memory doesn't exist
                return Ok(false);
            }
        }

        // Delegate to ingestion pipeline for atomic cleanup
        self.pipeline.delete(id)
    }

    /// Add multiple memories in a batch operation
    ///
    /// This is significantly faster than calling `add()` multiple times (10x+ improvement)
    /// because it uses:
    /// - Single transaction for all storage operations
    /// - Vector index locked once for all additions
    /// - Batched entity extraction with deduplication
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of MemoryInput to add
    ///
    /// # Returns
    ///
    /// BatchResult containing IDs of created memories and any errors
    ///
    /// # Performance
    ///
    /// Target: 1,000 memories in <500ms
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    /// use mnemefusion_core::types::MemoryInput;
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let inputs = vec![
    ///     MemoryInput::new("content 1".to_string(), vec![0.1; 384]),
    ///     MemoryInput::new("content 2".to_string(), vec![0.2; 384]),
    /// ];
    ///
    /// let result = engine.add_batch(inputs, None).unwrap();
    /// println!("Created {} memories", result.created_count);
    /// if result.has_errors() {
    ///     println!("Encountered {} errors", result.errors.len());
    /// }
    /// ```
    pub fn add_batch(
        &self,
        inputs: Vec<MemoryInput>,
        namespace: Option<&str>,
    ) -> Result<BatchResult> {
        // Validate all embeddings upfront
        for (index, input) in inputs.iter().enumerate() {
            if input.embedding.len() != self.config.embedding_dim {
                let mut result = BatchResult::new();
                result.errors.push(crate::types::BatchError::new(
                    index,
                    format!(
                        "Invalid embedding dimension: expected {}, got {}",
                        self.config.embedding_dim,
                        input.embedding.len()
                    ),
                ));
                return Ok(result);
            }
        }

        // Set namespace on all inputs if provided
        let mut inputs_with_ns = inputs;
        if let Some(ns) = namespace {
            for input in &mut inputs_with_ns {
                input.namespace = Some(ns.to_string());
            }
        }

        // Delegate to ingestion pipeline
        self.pipeline.add_batch(inputs_with_ns, None)
    }

    /// Delete multiple memories in a batch operation
    ///
    /// This is faster than calling `delete()` multiple times because it uses:
    /// - Single transaction for all storage operations
    /// - Batched entity cleanup
    ///
    /// # Arguments
    ///
    /// * `ids` - Vector of MemoryIds to delete
    /// * `namespace` - Optional namespace. If provided, only deletes memories in this namespace
    ///
    /// # Returns
    ///
    /// Number of memories actually deleted (may be less than input if some don't exist or are in wrong namespace)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// # let id1 = engine.add("test1".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// # let id2 = engine.add("test2".to_string(), vec![0.2; 384], None, None, None, None).unwrap();
    /// let ids = vec![id1, id2];
    /// let deleted_count = engine.delete_batch(ids, None).unwrap();
    /// println!("Deleted {} memories", deleted_count);
    /// ```
    pub fn delete_batch(&self, ids: Vec<MemoryId>, namespace: Option<&str>) -> Result<usize> {
        // If namespace is provided, filter IDs to only those in the namespace
        let ids_to_delete = if let Some(expected_ns) = namespace {
            let mut filtered_ids = Vec::new();
            for id in ids {
                if let Some(memory) = self.storage.get_memory(&id)? {
                    if memory.get_namespace() == expected_ns {
                        filtered_ids.push(id);
                    }
                }
            }
            filtered_ids
        } else {
            ids
        };

        // Delegate to ingestion pipeline
        self.pipeline.delete_batch(ids_to_delete)
    }

    /// Add a memory with automatic deduplication
    ///
    /// Uses content hash to detect duplicates. If identical content already exists,
    /// returns the existing memory ID without creating a duplicate.
    ///
    /// # Arguments
    ///
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// AddResult with created flag and ID (either new or existing)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // First add
    /// let result1 = engine.add_with_dedup(
    ///     "Meeting notes".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result1.created);
    ///
    /// // Second add with same content
    /// let result2 = engine.add_with_dedup(
    ///     "Meeting notes".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(!result2.created); // Duplicate detected
    /// assert_eq!(result1.id, result2.id); // Same ID returned
    /// ```
    pub fn add_with_dedup(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
    ) -> Result<AddResult> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            Memory::new_with_timestamp(content, embedding, ts)
        } else {
            Memory::new(content, embedding)
        };

        // Add metadata
        if let Some(meta) = metadata {
            for (key, value) in meta {
                memory.set_metadata(key, value);
            }
        }

        // Add source
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Set namespace if provided
        memory.set_namespace(namespace.unwrap_or(""));

        // Delegate to pipeline with deduplication
        self.pipeline.add_with_dedup(memory)
    }

    /// Upsert a memory by logical key
    ///
    /// If key exists: replaces content, embedding, and metadata
    /// If key doesn't exist: creates new memory and associates with key
    ///
    /// This is useful for updating facts that may change over time.
    ///
    /// # Arguments
    ///
    /// * `key` - Logical key (e.g., "user_profile:123", "doc:readme")
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// UpsertResult indicating whether memory was created or updated
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mnemefusion_core::{MemoryEngine, Config};
    ///
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    ///
    /// // First upsert - creates new
    /// let result1 = engine.upsert(
    ///     "user:profile",
    ///     "Alice likes hiking".to_string(),
    ///     embedding.clone(),
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result1.created);
    ///
    /// // Second upsert - updates existing
    /// let result2 = engine.upsert(
    ///     "user:profile",
    ///     "Alice likes hiking and photography".to_string(),
    ///     vec![0.2; 384],
    ///     None,
    ///     None,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// assert!(result2.updated);
    /// assert_eq!(result2.previous_content, Some("Alice likes hiking".to_string()));
    /// ```
    pub fn upsert(
        &self,
        key: &str,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
        namespace: Option<&str>,
    ) -> Result<UpsertResult> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let mut memory = if let Some(ts) = timestamp {
            Memory::new_with_timestamp(content, embedding, ts)
        } else {
            Memory::new(content, embedding)
        };

        // Add metadata
        if let Some(meta) = metadata {
            for (key, value) in meta {
                memory.set_metadata(key, value);
            }
        }

        // Add source
        if let Some(src) = source {
            memory.set_source(src)?;
        }

        // Set namespace if provided
        memory.set_namespace(namespace.unwrap_or(""));

        // Delegate to pipeline
        self.pipeline.upsert(key, memory)
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
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
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
    /// let results = engine.search(&query_embedding, 10, None, None).unwrap();
    /// for (memory, score) in results {
    ///     println!("Similarity: {:.3} - {}", score, memory.content);
    /// }
    /// ```
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<Vec<(Memory, f32)>> {
        // If filtering is needed, fetch more results (5x) and filter
        let needs_filtering =
            namespace.is_some() || (filters.is_some() && !filters.unwrap().is_empty());
        let fetch_k = if needs_filtering { top_k * 5 } else { top_k };

        // Search vector index
        let vector_results = {
            let index = self.vector_index.read().unwrap();
            index.search(query_embedding, fetch_k)?
        };

        // Retrieve full memory records using u64 lookup
        let mut results = Vec::with_capacity(vector_results.len());

        for vector_result in vector_results {
            // Look up memory using the u64 key from vector index
            let key = vector_result.id.to_u64();
            if let Some(memory) = self.storage.get_memory_by_u64(key)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }

                // Filter by metadata if provided
                if let Some(filter_list) = filters {
                    if !Self::memory_matches_filters(&memory, filter_list) {
                        continue;
                    }
                }

                results.push((memory, vector_result.similarity));

                // Stop if we have enough results after filtering
                if results.len() >= top_k {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Check if a memory matches all metadata filters
    fn memory_matches_filters(memory: &Memory, filters: &[MetadataFilter]) -> bool {
        for filter in filters {
            let value = memory.metadata.get(&filter.field).map(|s| s.as_str());
            if !filter.matches(value) {
                return false;
            }
        }
        true
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
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
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
    ///     10,
    ///     None,
    ///     None
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
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<(Memory, FusedResult)>)> {
        // Execute query using query planner
        let (intent, fused_results) =
            self.query_planner
                .query(query_text, query_embedding, limit, namespace, filters)?;

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
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
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
    /// let results = engine.get_range(week_ago, now, 100, None).unwrap();
    /// for (memory, timestamp) in results {
    ///     println!("{}: {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_range(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        // Fetch more results if filtering by namespace
        let fetch_limit = if namespace.is_some() {
            limit * 3
        } else {
            limit
        };
        let temporal_results = self.temporal_index.range_query(start, end, fetch_limit)?;

        // Retrieve and filter full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }
                results.push((memory, temp_result.timestamp));

                // Stop if we have enough results after filtering
                if results.len() >= limit {
                    break;
                }
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
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
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
    /// let recent = engine.get_recent(10, None).unwrap();
    /// println!("10 most recent memories:");
    /// for (memory, timestamp) in recent {
    ///     println!("  {} - {}", timestamp.as_unix_secs(), memory.content);
    /// }
    /// ```
    pub fn get_recent(
        &self,
        n: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        // Fetch more results if filtering by namespace
        let fetch_n = if namespace.is_some() { n * 3 } else { n };
        let temporal_results = self.temporal_index.recent(fetch_n)?;

        // Retrieve and filter full memory records
        let mut results = Vec::with_capacity(temporal_results.len());

        for temp_result in temporal_results {
            if let Some(memory) = self.storage.get_memory(&temp_result.id)? {
                // Filter by namespace if provided
                if let Some(ns) = namespace {
                    if memory.get_namespace() != ns {
                        continue;
                    }
                }
                results.push((memory, temp_result.timestamp));

                // Stop if we have enough results after filtering
                if results.len() >= n {
                    break;
                }
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
    /// # let id1 = engine.add("Cause".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// # let id2 = engine.add("Effect".to_string(), vec![0.2; 384], None, None, None, None).unwrap();
    /// engine.add_causal_link(&id1, &id2, 0.9, "id1 caused id2".to_string()).unwrap();
    /// ```
    pub fn add_causal_link(
        &self,
        cause: &MemoryId,
        effect: &MemoryId,
        confidence: f32,
        evidence: String,
    ) -> Result<()> {
        // Add the causal link to the graph
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.add_causal_link(cause, effect, confidence, evidence)?;
        }

        // Persist graph immediately for crash recovery
        // This ensures causal links are durable
        {
            let graph = self.graph_manager.read().unwrap();
            crate::graph::persist::save_graph(&graph, &self.storage)?;
        }

        Ok(())
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
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let causes = engine.get_causes(&id, 3).unwrap();
    /// for path in causes.paths {
    ///     println!("Found causal path with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_causes(
        &self,
        memory_id: &MemoryId,
        max_hops: usize,
    ) -> Result<CausalTraversalResult> {
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
    /// # let id = engine.add("Memory".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
    /// let effects = engine.get_effects(&id, 3).unwrap();
    /// for path in effects.paths {
    ///     println!("Found effect chain with {} steps (confidence: {})",
    ///              path.memories.len(), path.confidence);
    /// }
    /// ```
    pub fn get_effects(
        &self,
        memory_id: &MemoryId,
        max_hops: usize,
    ) -> Result<CausalTraversalResult> {
        let graph = self.graph_manager.read().unwrap();
        graph.get_effects(memory_id, max_hops)
    }

    // ========== Namespace Operations ==========

    /// List all namespaces in the database
    ///
    /// Returns a sorted list of all unique namespace strings, excluding the default namespace ("").
    ///
    /// # Performance
    ///
    /// O(n) where n = total memories. This scans all memories to extract namespaces.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let namespaces = engine.list_namespaces().unwrap();
    /// for ns in namespaces {
    ///     println!("Namespace: {}", ns);
    /// }
    /// ```
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        self.storage.list_namespaces()
    }

    /// Count memories in a specific namespace
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to count (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// Number of memories in the namespace
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let count = engine.count_namespace("user_123").unwrap();
    /// println!("User has {} memories", count);
    /// ```
    pub fn count_namespace(&self, namespace: &str) -> Result<usize> {
        self.storage.count_namespace(namespace)
    }

    /// Delete all memories in a namespace
    ///
    /// This is a convenience method that lists all memory IDs in the namespace
    /// and deletes them via the ingestion pipeline (ensuring proper cleanup of indexes).
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to delete (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    ///
    /// # Warning
    ///
    /// This operation cannot be undone. Use with caution.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let deleted = engine.delete_namespace("old_user").unwrap();
    /// println!("Deleted {} memories from namespace", deleted);
    /// ```
    pub fn delete_namespace(&self, namespace: &str) -> Result<usize> {
        // Get all memory IDs in this namespace
        let ids = self.storage.list_namespace_ids(namespace)?;

        // Delete via pipeline for proper cleanup
        self.pipeline.delete_batch(ids)
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
    /// # let id = engine.add("Alice met Bob".to_string(), vec![0.1; 384], None, None, None, None).unwrap();
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

    /// Create a scoped view for namespace-specific operations
    ///
    /// Returns a ScopedMemory that automatically applies the namespace to all operations.
    /// This provides a more ergonomic API when working with a single namespace.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to scope to (empty string "" for default namespace)
    ///
    /// # Returns
    ///
    /// A ScopedMemory view bound to this namespace
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// // Create scoped view for a user
    /// let user_memory = engine.scope("user_123");
    ///
    /// // All operations automatically use the namespace
    /// let id = user_memory.add("User note".to_string(), vec![0.1; 384], None, None, None).unwrap();
    /// let results = user_memory.search(&vec![0.1; 384], 10, None).unwrap();
    /// let count = user_memory.count().unwrap();
    /// user_memory.delete_all().unwrap();
    /// ```
    pub fn scope<S: Into<String>>(&self, namespace: S) -> ScopedMemory {
        ScopedMemory {
            engine: self,
            namespace: namespace.into(),
        }
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

/// Scoped memory view for namespace-specific operations
///
/// This wrapper provides an ergonomic API for working with a single namespace.
/// All operations automatically apply the namespace, eliminating the need to pass
/// it to every method call.
///
/// Created via `MemoryEngine::scope()`.
///
/// # Example
///
/// ```no_run
/// use mnemefusion_core::{MemoryEngine, Config};
///
/// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
/// // Create scoped view for a specific user
/// let user_memory = engine.scope("user_123");
///
/// // Add memory (automatically in user_123 namespace)
/// let id = user_memory.add(
///     "User note".to_string(),
///     vec![0.1; 384],
///     None,
///     None,
///     None,
/// ).unwrap();
///
/// // Search within namespace
/// let results = user_memory.search(&vec![0.1; 384], 10, None).unwrap();
///
/// // Count memories in namespace
/// println!("User has {} memories", user_memory.count().unwrap());
///
/// // Delete all memories in namespace
/// user_memory.delete_all().unwrap();
/// ```
pub struct ScopedMemory<'a> {
    engine: &'a MemoryEngine,
    namespace: String,
}

impl<'a> ScopedMemory<'a> {
    /// Add a memory to this namespace
    ///
    /// Equivalent to calling `engine.add(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `content` - Text content
    /// * `embedding` - Vector embedding
    /// * `metadata` - Optional metadata
    /// * `timestamp` - Optional custom timestamp
    /// * `source` - Optional source/provenance
    ///
    /// # Returns
    ///
    /// The ID of the created memory
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<MemoryId> {
        self.engine.add(
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Search for memories in this namespace
    ///
    /// Equivalent to calling `engine.search(..., Some(namespace), filters)`
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - Query vector
    /// * `top_k` - Maximum number of results
    /// * `filters` - Optional metadata filters
    ///
    /// # Returns
    ///
    /// Vector of (Memory, similarity_score) tuples
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<Vec<(Memory, f32)>> {
        self.engine
            .search(query_embedding, top_k, Some(&self.namespace), filters)
    }

    /// Delete a memory from this namespace
    ///
    /// Equivalent to calling `engine.delete(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to delete
    ///
    /// # Returns
    ///
    /// true if deleted, false if not found
    ///
    /// # Errors
    ///
    /// Returns `Error::NamespaceMismatch` if the memory exists but is in a different namespace
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        self.engine.delete(id, Some(&self.namespace))
    }

    /// Add multiple memories to this namespace in a batch
    ///
    /// Equivalent to calling `engine.add_batch(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of MemoryInput
    ///
    /// # Returns
    ///
    /// BatchResult with IDs and error information
    pub fn add_batch(&self, inputs: Vec<MemoryInput>) -> Result<BatchResult> {
        self.engine.add_batch(inputs, Some(&self.namespace))
    }

    /// Delete multiple memories from this namespace
    ///
    /// Equivalent to calling `engine.delete_batch(..., Some(namespace))`
    ///
    /// # Arguments
    ///
    /// * `ids` - Vector of memory IDs
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    pub fn delete_batch(&self, ids: Vec<MemoryId>) -> Result<usize> {
        self.engine.delete_batch(ids, Some(&self.namespace))
    }

    /// Add a memory with deduplication in this namespace
    ///
    /// Equivalent to calling `engine.add_with_dedup(..., Some(namespace))`
    pub fn add_with_dedup(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<AddResult> {
        self.engine.add_with_dedup(
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Upsert a memory in this namespace
    ///
    /// Equivalent to calling `engine.upsert(..., Some(namespace))`
    pub fn upsert(
        &self,
        key: &str,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
        source: Option<Source>,
    ) -> Result<UpsertResult> {
        self.engine.upsert(
            key,
            content,
            embedding,
            metadata,
            timestamp,
            source,
            Some(&self.namespace),
        )
    }

    /// Count memories in this namespace
    ///
    /// Equivalent to calling `engine.count_namespace(namespace)`
    ///
    /// # Returns
    ///
    /// Number of memories in the namespace
    pub fn count(&self) -> Result<usize> {
        self.engine.count_namespace(&self.namespace)
    }

    /// Delete all memories in this namespace
    ///
    /// Equivalent to calling `engine.delete_namespace(namespace)`
    ///
    /// # Returns
    ///
    /// Number of memories deleted
    ///
    /// # Warning
    ///
    /// This operation cannot be undone. Use with caution.
    pub fn delete_all(&self) -> Result<usize> {
        self.engine.delete_namespace(&self.namespace)
    }

    /// Multi-dimensional query within this namespace
    ///
    /// Equivalent to calling `engine.query(..., Some(namespace), filters)`
    ///
    /// # Arguments
    ///
    /// * `query_text` - Natural language query
    /// * `query_embedding` - Query vector
    /// * `limit` - Maximum number of results
    /// * `filters` - Optional metadata filters
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, results)
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<(Memory, FusedResult)>)> {
        self.engine.query(
            query_text,
            query_embedding,
            limit,
            Some(&self.namespace),
            filters,
        )
    }

    /// Get memories in time range within this namespace
    ///
    /// Equivalent to calling `engine.get_range(..., Some(namespace))`
    pub fn get_range(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<(Memory, Timestamp)>> {
        self.engine
            .get_range(start, end, limit, Some(&self.namespace))
    }

    /// Get recent memories within this namespace
    ///
    /// Equivalent to calling `engine.get_recent(..., Some(namespace))`
    pub fn get_recent(&self, n: usize) -> Result<Vec<(Memory, Timestamp)>> {
        self.engine.get_recent(n, Some(&self.namespace))
    }

    /// Get the namespace this view is scoped to
    pub fn namespace(&self) -> &str {
        &self.namespace
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

        let id = engine
            .add(content.clone(), embedding.clone(), None, None, None, None)
            .unwrap();

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

        let id = engine
            .add("test".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        let deleted = engine.delete(&id, None).unwrap();
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

        engine
            .add("test1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();
        assert_eq!(engine.count().unwrap(), 1);

        engine
            .add("test2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_memory_engine_list_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id1 = engine
            .add("test1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();
        let id2 = engine
            .add("test2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

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
            let id = engine
                .add(
                    "persistent".to_string(),
                    vec![0.5; 384],
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
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

    #[test]
    fn test_namespace_add_and_filter() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories to different namespaces
        let id1 = engine
            .add(
                "User 1 memory".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("user_1"),
            )
            .unwrap();

        let id2 = engine
            .add(
                "User 2 memory".to_string(),
                vec![0.2; 384],
                None,
                None,
                None,
                Some("user_2"),
            )
            .unwrap();

        let id3 = engine
            .add(
                "Default memory".to_string(),
                vec![0.3; 384],
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // Verify memories are in correct namespaces
        let mem1 = engine.get(&id1).unwrap().unwrap();
        assert_eq!(mem1.get_namespace(), "user_1");

        let mem2 = engine.get(&id2).unwrap().unwrap();
        assert_eq!(mem2.get_namespace(), "user_2");

        let mem3 = engine.get(&id3).unwrap().unwrap();
        assert_eq!(mem3.get_namespace(), "");

        // Test search with namespace filtering
        let query_embedding = vec![0.15; 384];

        // Search in user_1 namespace
        let results = engine
            .search(&query_embedding, 10, Some("user_1"), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id1);

        // Search in user_2 namespace
        let results = engine
            .search(&query_embedding, 10, Some("user_2"), None)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id2);

        // Search in default namespace
        let results = engine.search(&query_embedding, 10, Some(""), None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id3);

        // Search without namespace filter (should get all)
        let results = engine.search(&query_embedding, 10, None, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_namespace_delete_with_verification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memory to namespace
        let id = engine
            .add(
                "User memory".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("user_1"),
            )
            .unwrap();

        // Try to delete with wrong namespace - should fail
        let result = engine.delete(&id, Some("user_2"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::Error::NamespaceMismatch { .. }
        ));

        // Verify memory still exists
        assert!(engine.get(&id).unwrap().is_some());

        // Delete with correct namespace - should succeed
        let deleted = engine.delete(&id, Some("user_1")).unwrap();
        assert!(deleted);

        // Verify memory is gone
        assert!(engine.get(&id).unwrap().is_none());
    }

    #[test]
    fn test_namespace_management_methods() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories to different namespaces
        engine
            .add(
                "Memory 1".to_string(),
                vec![0.1; 384],
                None,
                None,
                None,
                Some("ns1"),
            )
            .unwrap();
        engine
            .add(
                "Memory 2".to_string(),
                vec![0.2; 384],
                None,
                None,
                None,
                Some("ns1"),
            )
            .unwrap();
        engine
            .add(
                "Memory 3".to_string(),
                vec![0.3; 384],
                None,
                None,
                None,
                Some("ns2"),
            )
            .unwrap();
        engine
            .add(
                "Memory 4".to_string(),
                vec![0.4; 384],
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // List namespaces
        let namespaces = engine.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);
        assert!(namespaces.contains(&"ns1".to_string()));
        assert!(namespaces.contains(&"ns2".to_string()));

        // Count in namespace
        assert_eq!(engine.count_namespace("ns1").unwrap(), 2);
        assert_eq!(engine.count_namespace("ns2").unwrap(), 1);
        assert_eq!(engine.count_namespace("").unwrap(), 1); // Default namespace

        // Delete entire namespace
        let deleted = engine.delete_namespace("ns1").unwrap();
        assert_eq!(deleted, 2);

        // Verify namespace is gone
        assert_eq!(engine.count_namespace("ns1").unwrap(), 0);
        let namespaces = engine.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 1);
        assert!(namespaces.contains(&"ns2".to_string()));

        // Total count should be 2 now
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_namespace_batch_operations() {
        use crate::types::MemoryInput;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create batch inputs
        let inputs = vec![
            MemoryInput::new("Memory 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Memory 2".to_string(), vec![0.2; 384]),
            MemoryInput::new("Memory 3".to_string(), vec![0.3; 384]),
        ];

        // Add batch with namespace
        let result = engine.add_batch(inputs, Some("batch_ns")).unwrap();
        assert_eq!(result.created_count, 3);
        assert!(result.is_success());

        // Verify all are in the namespace
        assert_eq!(engine.count_namespace("batch_ns").unwrap(), 3);

        // Batch delete with namespace filter
        let deleted = engine
            .delete_batch(result.ids.clone(), Some("batch_ns"))
            .unwrap();
        assert_eq!(deleted, 3);

        // Verify namespace is empty
        assert_eq!(engine.count_namespace("batch_ns").unwrap(), 0);
    }

    // ========== ScopedMemory Tests ==========

    #[test]
    fn test_scoped_memory_add_and_search() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create scoped view
        let scoped = engine.scope("user_123");

        // Add memory via scoped view
        let id = scoped
            .add(
                "Scoped memory".to_string(),
                vec![0.5; 384],
                None,
                None,
                None,
            )
            .unwrap();

        // Verify memory is in the namespace
        let memory = engine.get(&id).unwrap().unwrap();
        assert_eq!(memory.get_namespace(), "user_123");
        assert_eq!(memory.content, "Scoped memory");

        // Search via scoped view
        let results = scoped.search(&vec![0.5; 384], 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, id);

        // Verify namespace isolation
        assert_eq!(scoped.namespace(), "user_123");
    }

    #[test]
    fn test_scoped_memory_count_and_delete_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("user_456");

        // Add multiple memories
        scoped
            .add("Memory 1".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();
        scoped
            .add("Memory 2".to_string(), vec![0.2; 384], None, None, None)
            .unwrap();
        scoped
            .add("Memory 3".to_string(), vec![0.3; 384], None, None, None)
            .unwrap();

        // Count via scoped view
        assert_eq!(scoped.count().unwrap(), 3);

        // Total engine count should also be 3
        assert_eq!(engine.count().unwrap(), 3);

        // Delete all via scoped view
        let deleted = scoped.delete_all().unwrap();
        assert_eq!(deleted, 3);

        // Verify namespace is empty
        assert_eq!(scoped.count().unwrap(), 0);
        assert_eq!(engine.count().unwrap(), 0);
    }

    #[test]
    fn test_scoped_memory_isolation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Create two scoped views
        let scope1 = engine.scope("ns1");
        let scope2 = engine.scope("ns2");

        // Add memories to each namespace
        let id1 = scope1
            .add("NS1 memory".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();
        let id2 = scope2
            .add("NS2 memory".to_string(), vec![0.2; 384], None, None, None)
            .unwrap();

        // Each scope should only see its own memories
        assert_eq!(scope1.count().unwrap(), 1);
        assert_eq!(scope2.count().unwrap(), 1);

        // Search should be isolated
        let results1 = scope1.search(&vec![0.1; 384], 10, None).unwrap();
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0].0.id, id1);

        let results2 = scope2.search(&vec![0.2; 384], 10, None).unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].0.id, id2);

        // Delete from scope1 shouldn't affect scope2
        scope1.delete_all().unwrap();
        assert_eq!(scope1.count().unwrap(), 0);
        assert_eq!(scope2.count().unwrap(), 1);
    }

    #[test]
    fn test_scoped_memory_delete_with_verification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scope1 = engine.scope("ns1");
        let scope2 = engine.scope("ns2");

        // Add memory to ns1
        let id = scope1
            .add("NS1 memory".to_string(), vec![0.1; 384], None, None, None)
            .unwrap();

        // Try to delete from wrong namespace - should fail
        let result = scope2.delete(&id);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::Error::NamespaceMismatch { .. }
        ));

        // Verify memory still exists
        assert_eq!(scope1.count().unwrap(), 1);

        // Delete from correct namespace - should succeed
        let deleted = scope1.delete(&id).unwrap();
        assert!(deleted);
        assert_eq!(scope1.count().unwrap(), 0);
    }

    #[test]
    fn test_scoped_memory_batch_operations() {
        use crate::types::MemoryInput;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("batch_scope");

        // Add batch via scoped view
        let inputs = vec![
            MemoryInput::new("Batch 1".to_string(), vec![0.1; 384]),
            MemoryInput::new("Batch 2".to_string(), vec![0.2; 384]),
            MemoryInput::new("Batch 3".to_string(), vec![0.3; 384]),
        ];

        let result = scoped.add_batch(inputs).unwrap();
        assert_eq!(result.created_count, 3);
        assert!(result.is_success());

        // Verify count
        assert_eq!(scoped.count().unwrap(), 3);

        // Delete batch via scoped view
        let deleted = scoped.delete_batch(result.ids).unwrap();
        assert_eq!(deleted, 3);

        // Verify empty
        assert_eq!(scoped.count().unwrap(), 0);
    }

    #[test]
    fn test_search_with_metadata_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories with different metadata
        let mut mem1 = Memory::new("Event 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Event 2".to_string(), vec![0.11; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        engine
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem3 = Memory::new("Task 1".to_string(), vec![0.12; 384]);
        mem3.metadata.insert("type".to_string(), "task".to_string());
        mem3.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem3.content.clone(),
                mem3.embedding.clone(),
                Some(mem3.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        // Search with filter: type=event
        let filters = vec![MetadataFilter::eq("type", "event")];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|(m, _)| m.metadata.get("type").unwrap() == "event"));

        // Search with filter: priority=high
        let filters = vec![MetadataFilter::eq("priority", "high")];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .all(|(m, _)| m.metadata.get("priority").unwrap() == "high"));

        // Search with multiple filters: type=event AND priority=high
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        let results = engine
            .search(&vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Event 1");
    }

    #[test]
    fn test_query_with_metadata_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add memories with different metadata
        let mut mem1 = Memory::new("Important meeting".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        engine
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Casual meeting".to_string(), vec![0.11; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        engine
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
                None,
            )
            .unwrap();

        // Query with filter
        let filters = vec![MetadataFilter::eq("priority", "high")];
        let (_intent, results) = engine
            .query("meeting", &vec![0.1; 384], 10, None, Some(&filters))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Important meeting");
    }

    #[test]
    fn test_scoped_memory_with_filters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let scoped = engine.scope("user_123");

        // Add memories with different metadata to the namespace
        let mut mem1 = Memory::new("Event 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        scoped
            .add(
                mem1.content.clone(),
                mem1.embedding.clone(),
                Some(mem1.metadata.clone()),
                None,
                None,
            )
            .unwrap();

        let mut mem2 = Memory::new("Task 1".to_string(), vec![0.11; 384]);
        mem2.metadata.insert("type".to_string(), "task".to_string());
        scoped
            .add(
                mem2.content.clone(),
                mem2.embedding.clone(),
                Some(mem2.metadata.clone()),
                None,
                None,
            )
            .unwrap();

        // Search with filter in scoped view
        let filters = vec![MetadataFilter::eq("type", "event")];
        let results = scoped.search(&vec![0.1; 384], 10, Some(&filters)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.content, "Event 1");
    }
}
