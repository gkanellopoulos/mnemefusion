//! Vector index implementation using HNSW algorithm
//!
//! Wraps the usearch library to provide semantic similarity search
//! with persistence to the storage layer.

use crate::{storage::StorageEngine, types::MemoryId, Error, Result};
use std::sync::Arc;

/// Configuration for HNSW vector index
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Vector dimension (must match embeddings)
    pub dimension: usize,

    /// Number of bi-directional links per node (HNSW M parameter)
    /// Higher values improve recall but increase memory usage
    /// Typical range: 12-48, default: 16
    pub connectivity: usize,

    /// Size of dynamic candidate list during construction (ef_construction)
    /// Higher values improve index quality but slow construction
    /// Typical range: 100-500, default: 128
    pub expansion_add: usize,

    /// Size of dynamic candidate list during search (ef_search)
    /// Higher values improve recall but slow search
    /// Can be adjusted at query time
    /// Typical range: 100-500, default: 64
    pub expansion_search: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

/// Result from vector similarity search
#[derive(Debug, Clone)]
pub struct VectorResult {
    /// Memory ID
    pub id: MemoryId,

    /// Cosine similarity score (0.0 to 1.0)
    /// Higher is more similar
    pub similarity: f32,
}

/// Vector index for semantic similarity search
///
/// Uses HNSW (Hierarchical Navigable Small World) algorithm for
/// efficient approximate nearest neighbor search.
pub struct VectorIndex {
    index: usearch::Index,
    config: VectorIndexConfig,
    storage: Arc<StorageEngine>,
    count: usize, // Track number of vectors for debugging
}

impl VectorIndex {
    /// Create a new vector index
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration
    /// * `storage` - Storage engine for persistence
    pub fn new(config: VectorIndexConfig, storage: Arc<StorageEngine>) -> Result<Self> {
        // Create usearch index with configuration
        let options = usearch::IndexOptions {
            dimensions: config.dimension,
            metric: usearch::MetricKind::Cos, // Cosine similarity
            quantization: usearch::ScalarKind::F32, // No quantization for now
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false, // Single vector per key
        };

        let index = usearch::new_index(&options)
            .map_err(|e| Error::VectorIndex(format!("Failed to create index: {}", e)))?;

        // Reserve initial capacity (helps with Windows initialization)
        index
            .reserve(1000)
            .map_err(|e| Error::VectorIndex(format!("Failed to reserve capacity: {}", e)))?;

        Ok(Self {
            index,
            config,
            storage,
            count: 0,
        })
    }

    /// Add a vector to the index
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID
    /// * `embedding` - Vector embedding (must match configured dimension)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Embedding dimension doesn't match configuration
    /// - Index operation fails
    pub fn add(&mut self, id: MemoryId, embedding: &[f32]) -> Result<()> {
        // Validate dimension
        if embedding.len() != self.config.dimension {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.dimension,
                got: embedding.len(),
            });
        }

        // Convert MemoryId to u64 key for usearch
        let key = id.to_u64();

        // Add to index
        self.index
            .add(key, embedding)
            .map_err(|e| Error::VectorIndex(format!("Failed to add vector: {}", e)))?;

        self.count += 1;

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of results sorted by similarity (highest first)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Query dimension doesn't match configuration
    /// - Search operation fails
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorResult>> {
        // Validate dimension
        if query.len() != self.config.dimension {
            return Err(Error::InvalidEmbeddingDimension {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        // Handle empty index
        if self.count == 0 {
            return Ok(Vec::new());
        }

        // Search index
        let matches = self
            .index
            .search(query, top_k)
            .map_err(|e| Error::VectorIndex(format!("Search failed: {}", e)))?;

        // Convert results
        // matches.keys and matches.distances are vectors of the same length
        let count = matches.keys.len();
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let key = matches.keys[i];
            let distance = matches.distances[i];

            // Convert u64 back to MemoryId
            // Note: This is a partial reconstruction. The full UUID is stored
            // in the database and we'll fetch it during result resolution.
            let id = MemoryId::from_u64(key);

            // Convert distance to similarity
            // For cosine distance: similarity = 1 - distance
            // usearch returns distance in range [0, 2] for cosine
            let similarity = (1.0 - distance).max(0.0).min(1.0);

            results.push(VectorResult { id, similarity });
        }

        Ok(results)
    }

    /// Remove a vector from the index
    ///
    /// # Arguments
    ///
    /// * `id` - Memory ID to remove
    pub fn remove(&mut self, id: &MemoryId) -> Result<()> {
        let key = id.to_u64();

        self.index
            .remove(key)
            .map_err(|e| Error::VectorIndex(format!("Failed to remove vector: {}", e)))?;

        self.count = self.count.saturating_sub(1);

        Ok(())
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Save the index to storage
    ///
    /// Serializes the entire HNSW index to the METADATA_TABLE
    /// for persistence across restarts.
    pub fn save(&self) -> Result<()> {
        // Empty index doesn't need saving
        if self.count == 0 {
            return Ok(());
        }

        // Estimate buffer size: vectors + graph structure
        // Formula: (dimensions * 4 bytes) + HNSW overhead per vector (~1KB)
        let bytes_per_vector = (self.config.dimension * 4) + 1024;
        let estimated_size = self.count * bytes_per_vector + 100_000; // 100KB base overhead

        // Try with increasing buffer sizes
        for attempt in 0..3 {
            let buffer_size = estimated_size * (1 << attempt); // 1x, 2x, 4x
            let mut buffer = vec![0u8; buffer_size];

            match self.index.save_to_buffer(&mut buffer) {
                Ok(_) => {
                    // Store in storage engine
                    self.storage.store_vector_index(&buffer)?;
                    return Ok(());
                }
                Err(_) if attempt < 2 => {
                    // Try with larger buffer
                    continue;
                }
                Err(e) => {
                    return Err(Error::VectorIndex(format!(
                        "Failed to serialize index: {}",
                        e
                    )));
                }
            }
        }

        Err(Error::VectorIndex(format!(
            "Failed to serialize index after multiple attempts. Index size: {} vectors. \
                 This may indicate the index is too large or corrupted.",
            self.count
        )))
    }

    /// Load the index from storage
    ///
    /// Deserializes the HNSW index from the METADATA_TABLE.
    /// This should be called after creating a new VectorIndex to restore
    /// previously saved state.
    pub fn load(&mut self) -> Result<()> {
        // Check if index exists in storage
        match self.storage.load_vector_index()? {
            Some(buffer) => {
                // Validate buffer is not empty
                if buffer.is_empty() {
                    return Err(Error::DatabaseCorruption(
                        "Vector index buffer is empty".to_string(),
                    ));
                }

                // Validate minimum buffer size (usearch has minimum overhead)
                const MIN_INDEX_SIZE: usize = 100; // Conservative minimum for usearch metadata
                if buffer.len() < MIN_INDEX_SIZE {
                    return Err(Error::DatabaseCorruption(format!(
                        "Vector index buffer too small: {} bytes",
                        buffer.len()
                    )));
                }

                // Load from buffer
                self.index
                    .load_from_buffer(&buffer)
                    .map_err(|e| Error::VectorIndex(format!("Failed to load index: {}", e)))?;

                // Update count
                self.count = self.index.size();

                // Validate loaded index
                self.validate()?;

                Ok(())
            }
            None => {
                // No index to load, this is a new database
                Ok(())
            }
        }
    }

    /// Validate vector index integrity
    ///
    /// Checks that the index is in a valid state:
    /// - Size is consistent
    /// - Configuration matches expectations
    /// - Index can perform basic operations
    pub fn validate(&self) -> Result<()> {
        // Check that usearch size matches our count
        let usearch_size = self.index.size();
        if usearch_size != self.count {
            return Err(Error::DatabaseCorruption(format!(
                "Vector index size mismatch: internal count = {}, usearch size = {}",
                self.count, usearch_size
            )));
        }

        // Validate configuration consistency
        let index_dimensions = self.index.dimensions();
        if index_dimensions != self.config.dimension {
            return Err(Error::DatabaseCorruption(format!(
                "Vector index dimension mismatch: config = {}, index = {}",
                self.config.dimension, index_dimensions
            )));
        }

        // Additional validation: check that the index can perform a basic operation
        // Try a search with a zero vector (should not crash even if no results)
        if self.count > 0 {
            let zero_vector = vec![0.0f32; self.config.dimension];
            let _ = self.index.search(&zero_vector, 1).map_err(|e| {
                Error::DatabaseCorruption(format!("Vector index failed basic search test: {}", e))
            })?;
        }

        Ok(())
    }

    /// Get the index configuration
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }

    /// Update search expansion parameter (ef_search)
    ///
    /// This can be adjusted at query time to trade off between
    /// speed and recall. Higher values = better recall, slower search.
    pub fn set_expansion_search(&mut self, expansion: usize) {
        self.config.expansion_search = expansion;
        // Note: usearch doesn't expose a way to update ef_search after creation
        // This will be used for future index creations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageEngine;
    use tempfile::tempdir;

    fn create_test_index() -> (VectorIndex, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&path).unwrap());

        let config = VectorIndexConfig {
            dimension: 3, // Small dimension for testing
            ..Default::default()
        };

        let index = VectorIndex::new(config, storage).unwrap();
        (index, dir)
    }

    #[test]
    fn test_vector_index_create() {
        let (index, _dir) = create_test_index();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_vector_index_add() {
        let (mut index, _dir) = create_test_index();

        let id = MemoryId::new();
        let embedding = vec![0.1, 0.2, 0.3];

        index.add(id, &embedding).unwrap();
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_vector_index_add_wrong_dimension() {
        let (mut index, _dir) = create_test_index();

        let id = MemoryId::new();
        let embedding = vec![0.1, 0.2]; // Wrong dimension

        let result = index.add(id, &embedding);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_search() {
        let (mut index, _dir) = create_test_index();

        // Add some vectors
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        let id3 = MemoryId::new();

        index.add(id1.clone(), &[1.0, 0.0, 0.0]).unwrap();
        index.add(id2.clone(), &[0.0, 1.0, 0.0]).unwrap();
        index.add(id3.clone(), &[0.0, 0.0, 1.0]).unwrap();

        // Search for vector similar to first one
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // First result should be most similar
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn test_vector_index_search_empty() {
        let (index, _dir) = create_test_index();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_vector_index_remove() {
        let (mut index, _dir) = create_test_index();

        let id = MemoryId::new();
        index.add(id.clone(), &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);

        index.remove(&id).unwrap();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_vector_index_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        // Create index, add vectors, save
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            index.add(id1.clone(), &[1.0, 0.0, 0.0]).unwrap();
            index.add(id2.clone(), &[0.0, 1.0, 0.0]).unwrap();

            index.save().unwrap();
        }

        // Reopen and load index
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            index.load().unwrap();

            assert_eq!(index.len(), 2);

            // Verify we can search
            let results = index.search(&[0.9, 0.1, 0.0], 2).unwrap();
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn test_vector_index_multiple_additions() {
        let (mut index, _dir) = create_test_index();

        // Add 100 vectors
        for i in 0..100 {
            let id = MemoryId::new();
            let val = (i as f32) / 100.0;
            index.add(id, &[val, 1.0 - val, 0.5]).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search should return results
        let results = index.search(&[0.5, 0.5, 0.5], 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    // Validation tests for Task 5: Vector index integrity validation

    #[test]
    fn test_vector_index_validate_empty() {
        let (index, _dir) = create_test_index();

        // Empty index should validate successfully
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_vector_index_validate_with_data() {
        let (mut index, _dir) = create_test_index();

        // Add some vectors
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        index.add(id1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(id2, &[0.0, 1.0, 0.0]).unwrap();

        // Should validate successfully
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_vector_index_validate_after_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let id1 = MemoryId::new();
        let id2 = MemoryId::new();

        // Create index, add vectors, save
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            index.add(id1, &[1.0, 0.0, 0.0]).unwrap();
            index.add(id2, &[0.0, 1.0, 0.0]).unwrap();

            index.save().unwrap();
        }

        // Reopen and load index
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            index.load().unwrap();

            // Validation should pass
            assert!(index.validate().is_ok());
            assert_eq!(index.len(), 2);
        }
    }

    #[test]
    fn test_vector_index_corrupted_buffer_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        // Create database and store empty buffer (corrupted)
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            storage.store_vector_index(&[]).unwrap();
        }

        // Try to load - should fail
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            let result = index.load();
            assert!(result.is_err());

            if let Err(err) = result {
                assert!(matches!(err, Error::DatabaseCorruption(_)));
            }
        }
    }

    #[test]
    fn test_vector_index_corrupted_buffer_too_small() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        // Create database and store tiny corrupted buffer
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let tiny_buffer = vec![0u8; 50]; // Too small
            storage.store_vector_index(&tiny_buffer).unwrap();
        }

        // Try to load - should fail
        {
            let storage = Arc::new(StorageEngine::open(&path).unwrap());
            let config = VectorIndexConfig {
                dimension: 3,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config, storage).unwrap();

            let result = index.load();
            assert!(result.is_err());

            if let Err(err) = result {
                assert!(matches!(err, Error::DatabaseCorruption(_)));
            }
        }
    }
}
