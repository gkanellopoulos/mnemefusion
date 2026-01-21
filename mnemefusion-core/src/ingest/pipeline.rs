//! Ingestion pipeline for coordinating memory storage across all dimensions
//!
//! The ingestion pipeline ensures that memories are indexed across all dimensions
//! atomically. If any indexing step fails, changes are rolled back to maintain
//! consistency.

use crate::{
    error::Result,
    graph::GraphManager,
    index::{TemporalIndex, VectorIndex},
    ingest::{EntityExtractor, SimpleEntityExtractor},
    storage::StorageEngine,
    types::{Entity, Memory, MemoryId},
};
use std::sync::{Arc, RwLock};

/// Coordinates memory ingestion across all dimensions
///
/// The IngestionPipeline ensures that all dimension indexes are updated
/// atomically when adding or deleting memories. This prevents partial
/// state if any indexing operation fails.
pub struct IngestionPipeline {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    entity_extractor: SimpleEntityExtractor,
    entity_extraction_enabled: bool,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage engine for persistent data
    /// * `vector_index` - Vector index for semantic similarity
    /// * `temporal_index` - Temporal index for time-based queries
    /// * `graph_manager` - Graph manager for causal and entity relationships
    /// * `entity_extraction_enabled` - Whether to automatically extract entities
    pub fn new(
        storage: Arc<StorageEngine>,
        vector_index: Arc<RwLock<VectorIndex>>,
        temporal_index: Arc<TemporalIndex>,
        graph_manager: Arc<RwLock<GraphManager>>,
        entity_extraction_enabled: bool,
    ) -> Self {
        Self {
            storage,
            vector_index,
            temporal_index,
            graph_manager,
            entity_extractor: SimpleEntityExtractor::new(),
            entity_extraction_enabled,
        }
    }

    /// Add a memory and index it across all dimensions
    ///
    /// This operation is atomic - if any indexing step fails, all changes
    /// are rolled back and an error is returned.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory to add (must have valid embedding)
    ///
    /// # Returns
    ///
    /// The MemoryId of the added memory
    ///
    /// # Errors
    ///
    /// Returns error if any indexing step fails. The database state will be
    /// consistent - no partial indexing will occur.
    pub fn add(&self, memory: Memory) -> Result<MemoryId> {
        let id = memory.id.clone();
        let timestamp = memory.created_at;

        // Step 1: Store memory (if this fails, nothing else happens)
        self.storage.store_memory(&memory)?;

        // Step 2: Add to vector index (rollback: delete from storage)
        if let Err(e) = self.add_to_vector_index(&id, &memory.embedding) {
            // Rollback: remove from storage
            let _ = self.storage.delete_memory(&id);
            return Err(e);
        }

        // Step 3: Add to temporal index (rollback: delete from storage + vector)
        if let Err(e) = self.temporal_index.add(&id, timestamp) {
            // Rollback: remove from storage and vector index
            let _ = self.storage.delete_memory(&id);
            let _ = self.remove_from_vector_index(&id);
            return Err(e);
        }

        // Step 4: Extract and link entities (if enabled)
        if self.entity_extraction_enabled {
            if let Err(e) = self.extract_and_link_entities(&id, &memory.content) {
                // Rollback: remove from all indexes
                let _ = self.storage.delete_memory(&id);
                let _ = self.remove_from_vector_index(&id);
                let _ = self.temporal_index.remove(&id);
                return Err(e);
            }
        }

        Ok(id)
    }

    /// Delete a memory and clean up all indexes
    ///
    /// This removes the memory from:
    /// - Storage
    /// - Vector index
    /// - Temporal index
    /// - Entity graph
    /// - Causal graph
    ///
    /// Orphaned entities (entities with no remaining mentions) are also removed.
    ///
    /// # Arguments
    ///
    /// * `id` - The MemoryId to delete
    ///
    /// # Returns
    ///
    /// true if the memory was deleted, false if it didn't exist
    ///
    /// # Errors
    ///
    /// Returns error if any deletion step fails. The database may be in an
    /// inconsistent state if an error occurs during cleanup.
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        // Check if memory exists
        let _memory = match self.storage.get_memory(id)? {
            Some(mem) => mem,
            None => return Ok(false), // Memory doesn't exist
        };

        // Get entities linked to this memory BEFORE deleting
        let entity_ids = {
            let graph = self.graph_manager.read().unwrap();
            graph.get_memory_entities(id)
        };

        // Step 1: Delete from storage
        self.storage.delete_memory(id)?;

        // Step 2: Remove from vector index
        // Ignore errors - index might not have the entry
        let _ = self.remove_from_vector_index(id);

        // Step 3: Remove from temporal index
        // Ignore errors - index might not have the entry
        let _ = self.temporal_index.remove(id);

        // Step 4: Remove from entity graph and clean up orphaned entities
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.remove_memory_from_entity_graph(id);
        }

        // Check each entity to see if it's now orphaned
        for entity_id in entity_ids {
            if let Some(mut entity) = self.storage.get_entity(&entity_id)? {
                // Decrement mention count
                entity.decrement_mentions();

                if entity.mention_count == 0 {
                    // Entity is orphaned, remove it
                    self.storage.delete_entity(&entity_id)?;

                    // Remove from entity graph
                    let mut graph = self.graph_manager.write().unwrap();
                    graph.remove_entity_from_graph(&entity_id);
                } else {
                    // Entity still has mentions, just update the count
                    self.storage.store_entity(&entity)?;
                }
            }
        }

        // Step 5: Remove from causal graph
        {
            let mut graph = self.graph_manager.write().unwrap();
            graph.remove_memory_from_causal_graph(id);
        }

        Ok(true)
    }

    /// Add memory to vector index
    fn add_to_vector_index(&self, id: &MemoryId, embedding: &[f32]) -> Result<()> {
        let mut index = self.vector_index.write().unwrap();
        index.add(id.clone(), embedding)
    }

    /// Remove memory from vector index
    fn remove_from_vector_index(&self, id: &MemoryId) -> Result<()> {
        let mut index = self.vector_index.write().unwrap();
        index.remove(id)
    }

    /// Extract entities from content and link to memory
    fn extract_and_link_entities(&self, memory_id: &MemoryId, content: &str) -> Result<()> {
        // Extract entity names
        let entity_names = self.entity_extractor.extract(content)?;

        // Deduplicate entity names (case-insensitive)
        let mut seen = std::collections::HashSet::new();
        let unique_entities: Vec<&String> = entity_names
            .iter()
            .filter(|name| seen.insert(name.to_lowercase()))
            .collect();

        // Process each unique extracted entity
        for entity_name in unique_entities {
            // Check if entity already exists
            let entity = match self.storage.find_entity_by_name(entity_name)? {
                Some(mut existing_entity) => {
                    // Entity exists, increment mention count
                    existing_entity.increment_mentions();
                    self.storage.store_entity(&existing_entity)?;
                    existing_entity
                }
                None => {
                    // Create new entity (starts with mention_count = 0)
                    let mut new_entity = Entity::new(entity_name.clone());
                    new_entity.increment_mentions();
                    self.storage.store_entity(&new_entity)?;
                    new_entity
                }
            };

            // Link memory to entity in graph
            let mut graph = self.graph_manager.write().unwrap();
            graph.link_memory_to_entity(memory_id, &entity.id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::GraphManager,
        index::{TemporalIndex, VectorIndex, VectorIndexConfig},
        storage::StorageEngine,
        types::{Memory, Timestamp},
    };
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_pipeline() -> (IngestionPipeline, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let storage = Arc::new(StorageEngine::open(&path).unwrap());

        let vector_config = VectorIndexConfig {
            dimension: 384,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        };
        let vector_index = Arc::new(RwLock::new(
            VectorIndex::new(vector_config, Arc::clone(&storage)).unwrap(),
        ));

        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));
        let graph_manager = Arc::new(RwLock::new(GraphManager::new()));

        let pipeline = IngestionPipeline::new(
            storage,
            vector_index,
            temporal_index,
            graph_manager,
            true, // Enable entity extraction
        );

        (pipeline, dir)
    }

    #[test]
    fn test_pipeline_add_and_retrieve() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        // Add memory
        let returned_id = pipeline.add(memory).unwrap();
        assert_eq!(id, returned_id);

        // Verify it's in storage
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_pipeline_add_indexes_all_dimensions() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        let timestamp = memory.created_at;

        pipeline.add(memory).unwrap();

        // Verify vector index
        {
            let index = pipeline.vector_index.read().unwrap();
            let results = index.search(&vec![0.1; 384], 10).unwrap();
            assert!(!results.is_empty());
        }

        // Verify temporal index (use wider time range)
        let start = Timestamp::from_unix_secs(timestamp.as_unix_secs() - 1.0);
        let end = Timestamp::from_unix_secs(timestamp.as_unix_secs() + 1.0);
        let temporal_results = pipeline
            .temporal_index
            .range_query(start, end, 10)
            .unwrap();
        assert_eq!(temporal_results.len(), 1);
        assert_eq!(temporal_results[0].id, id);
    }

    #[test]
    fn test_pipeline_add_extracts_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Alice met Bob in Paris".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        pipeline.add(memory).unwrap();

        // Check that entities were created
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());

        let bob = pipeline.storage.find_entity_by_name("Bob").unwrap();
        assert!(bob.is_some());

        let paris = pipeline.storage.find_entity_by_name("Paris").unwrap();
        assert!(paris.is_some());

        // Check that entities are linked in graph
        let graph = pipeline.graph_manager.read().unwrap();
        let entities = graph.get_memory_entities(&id);
        assert_eq!(entities.len(), 3);
    }

    #[test]
    fn test_pipeline_delete_removes_from_all_indexes() {
        let (pipeline, _dir) = create_test_pipeline();

        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();

        pipeline.add(memory).unwrap();

        // Delete the memory
        let deleted = pipeline.delete(&id).unwrap();
        assert!(deleted);

        // Verify it's removed from storage
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_none());

        // Verify it's removed from vector index (search won't find it)
        {
            let index = pipeline.vector_index.read().unwrap();
            let results = index.search(&vec![0.1; 384], 10).unwrap();
            assert!(results.is_empty() || !results.iter().any(|r| r.id == id));
        }

        // Verify it's removed from entity graph
        let graph = pipeline.graph_manager.read().unwrap();
        let entities = graph.get_memory_entities(&id);
        assert!(entities.is_empty());
    }

    #[test]
    fn test_pipeline_delete_cleans_orphaned_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add memory with entity
        let memory = Memory::new("Alice is here".to_string(), vec![0.1; 384]);
        let id = memory.id.clone();
        pipeline.add(memory).unwrap();

        // Verify entity exists
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 1);

        // Delete the memory
        pipeline.delete(&id).unwrap();

        // Verify entity is removed (orphaned)
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());
    }

    #[test]
    fn test_pipeline_delete_keeps_non_orphaned_entities() {
        let (pipeline, _dir) = create_test_pipeline();

        // Add two memories mentioning Alice
        let memory1 = Memory::new("Alice is here".to_string(), vec![0.1; 384]);
        let id1 = memory1.id.clone();
        pipeline.add(memory1).unwrap();

        let memory2 = Memory::new("Alice is there too".to_string(), vec![0.2; 384]);
        let id2 = memory2.id.clone();
        pipeline.add(memory2).unwrap();

        // Verify Alice has 2 mentions
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 2);

        // Delete first memory
        pipeline.delete(&id1).unwrap();

        // Verify Alice still exists with 1 mention
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_some());
        let alice = alice.unwrap();
        assert_eq!(alice.mention_count, 1);

        // Delete second memory
        pipeline.delete(&id2).unwrap();

        // Now Alice should be removed
        let alice = pipeline.storage.find_entity_by_name("Alice").unwrap();
        assert!(alice.is_none());
    }

    #[test]
    fn test_pipeline_delete_nonexistent() {
        let (pipeline, _dir) = create_test_pipeline();

        let id = MemoryId::new();
        let deleted = pipeline.delete(&id).unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_pipeline_rollback_on_vector_index_failure() {
        let (pipeline, _dir) = create_test_pipeline();

        // Create memory with wrong dimension to trigger vector index error
        let memory = Memory::new("Test".to_string(), vec![0.1; 512]);
        let id = memory.id.clone();

        let result = pipeline.add(memory);
        assert!(result.is_err());

        // Verify memory was NOT stored (rollback worked)
        let retrieved = pipeline.storage.get_memory(&id).unwrap();
        assert!(retrieved.is_none());
    }
}
