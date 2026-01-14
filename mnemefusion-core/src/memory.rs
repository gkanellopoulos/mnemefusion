//! Memory engine - main API entry point
//!
//! The MemoryEngine struct provides the primary interface for interacting
//! with a MnemeFusion database.

use crate::{
    config::Config,
    error::{Error, Result},
    storage::StorageEngine,
    types::{Memory, MemoryId, Timestamp},
};
use std::collections::HashMap;
use std::path::Path;

/// Main memory engine interface
///
/// This is the primary entry point for all MnemeFusion operations.
/// It coordinates storage, indexing, and retrieval across all dimensions.
pub struct MemoryEngine {
    storage: StorageEngine,
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
        let storage = StorageEngine::open(path)?;

        Ok(Self { storage, config })
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
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{MemoryEngine, Config};
    /// # let engine = MemoryEngine::open("./test.mfdb", Config::default()).unwrap();
    /// let embedding = vec![0.1; 384];
    /// let id = engine.add(
    ///     "Meeting scheduled for next week".to_string(),
    ///     embedding,
    ///     None,
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn add(
        &self,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
        timestamp: Option<Timestamp>,
    ) -> Result<MemoryId> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }

        // Create memory
        let memory = if let Some(ts) = timestamp {
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

        let id = memory.id.clone();

        // Store memory
        self.storage.store_memory(&memory)?;

        // TODO (Sprint 2): Add to vector index
        // TODO (Sprint 3): Add to temporal index
        // TODO (Sprint 5): Extract and link entities

        Ok(id)
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
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None).unwrap();
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
    /// # let id = engine.add("test".to_string(), vec![0.1; 384], None, None).unwrap();
    /// let deleted = engine.delete(&id).unwrap();
    /// assert!(deleted);
    /// ```
    pub fn delete(&self, id: &MemoryId) -> Result<bool> {
        // TODO (Sprint 6): Remove from all indexes
        self.storage.delete_memory(id)
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

    /// Close the database
    ///
    /// This ensures all data is flushed to disk. While not strictly necessary
    /// (redb handles this automatically), it's good practice to call this
    /// explicitly when you're done.
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
        // TODO (Sprint 2+): Save all indexes
        // For now, redb handles persistence automatically
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

        let id = engine.add(content.clone(), embedding.clone(), None, None).unwrap();

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

        let id = engine.add("test".to_string(), vec![0.1; 384], None, None).unwrap();

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

        engine.add("test1".to_string(), vec![0.1; 384], None, None).unwrap();
        assert_eq!(engine.count().unwrap(), 1);

        engine.add("test2".to_string(), vec![0.2; 384], None, None).unwrap();
        assert_eq!(engine.count().unwrap(), 2);
    }

    #[test]
    fn test_memory_engine_list_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let id1 = engine.add("test1".to_string(), vec![0.1; 384], None, None).unwrap();
        let id2 = engine.add("test2".to_string(), vec![0.2; 384], None, None).unwrap();

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
            let id = engine.add("persistent".to_string(), vec![0.5; 384], None, None).unwrap();
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
