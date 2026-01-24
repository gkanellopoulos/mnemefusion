//! Configuration types for MnemeFusion
//!
//! This module defines configuration options for the memory engine.

/// Configuration for the MnemeFusion memory engine
#[derive(Debug, Clone)]
pub struct Config {
    /// Dimension of embedding vectors
    pub embedding_dim: usize,

    /// Half-life for temporal decay in hours
    /// After this many hours, a memory's temporal relevance score is halved
    pub temporal_decay_hours: f32,

    /// Maximum number of hops for causal graph traversal
    pub causal_max_hops: usize,

    /// Enable automatic entity extraction
    pub entity_extraction_enabled: bool,

    /// Minimum confidence threshold for causal links (0.0 to 1.0)
    pub causal_min_confidence: f32,

    /// HNSW M parameter (connectivity)
    /// Higher values = better recall, more memory
    pub hnsw_m: usize,

    /// HNSW ef_construction parameter
    /// Higher values = better index quality, slower construction
    pub hnsw_ef_construction: usize,

    /// HNSW ef_search parameter
    /// Higher values = better recall, slower search
    pub hnsw_ef_search: usize,

    /// Metadata fields to index for efficient filtering
    /// These fields will have dedicated indexes for fast lookup
    /// Example: vec!["type".to_string(), "category".to_string(), "priority".to_string()]
    pub indexed_metadata: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_dim: 384, // Default for all-MiniLM-L6-v2
            temporal_decay_hours: 168.0, // 1 week
            causal_max_hops: 3,
            entity_extraction_enabled: true,
            causal_min_confidence: 0.5,
            hnsw_m: 16,
            hnsw_ef_construction: 128,
            hnsw_ef_search: 64,
            indexed_metadata: Vec::new(), // No indexed fields by default
        }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimension
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set the temporal decay half-life in hours
    pub fn with_temporal_decay_hours(mut self, hours: f32) -> Self {
        self.temporal_decay_hours = hours;
        self
    }

    /// Set the maximum causal traversal hops
    pub fn with_causal_max_hops(mut self, hops: usize) -> Self {
        self.causal_max_hops = hops;
        self
    }

    /// Enable or disable entity extraction
    pub fn with_entity_extraction(mut self, enabled: bool) -> Self {
        self.entity_extraction_enabled = enabled;
        self
    }

    /// Set HNSW parameters for vector index
    pub fn with_hnsw_params(mut self, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        self.hnsw_m = m;
        self.hnsw_ef_construction = ef_construction;
        self.hnsw_ef_search = ef_search;
        self
    }

    /// Set metadata fields to index for efficient filtering
    pub fn with_indexed_metadata(mut self, fields: Vec<String>) -> Self {
        self.indexed_metadata = fields;
        self
    }

    /// Add a metadata field to the indexed set
    pub fn add_indexed_field(mut self, field: impl Into<String>) -> Self {
        self.indexed_metadata.push(field.into());
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), crate::Error> {
        if self.embedding_dim == 0 {
            return Err(crate::Error::Configuration(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        if self.temporal_decay_hours <= 0.0 {
            return Err(crate::Error::Configuration(
                "temporal_decay_hours must be positive".to_string(),
            ));
        }

        if self.causal_max_hops == 0 {
            return Err(crate::Error::Configuration(
                "causal_max_hops must be greater than 0".to_string(),
            ));
        }

        if self.causal_min_confidence < 0.0 || self.causal_min_confidence > 1.0 {
            return Err(crate::Error::Configuration(
                "causal_min_confidence must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.hnsw_m == 0 {
            return Err(crate::Error::Configuration(
                "hnsw_m must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.temporal_decay_hours, 168.0);
        assert_eq!(config.causal_max_hops, 3);
        assert!(config.entity_extraction_enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = Config::new()
            .with_embedding_dim(512)
            .with_temporal_decay_hours(336.0)
            .with_causal_max_hops(5);

        assert_eq!(config.embedding_dim, 512);
        assert_eq!(config.temporal_decay_hours, 336.0);
        assert_eq!(config.causal_max_hops, 5);
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());

        let mut bad_config = Config::default();
        bad_config.embedding_dim = 0;
        assert!(bad_config.validate().is_err());

        let mut bad_config = Config::default();
        bad_config.temporal_decay_hours = -1.0;
        assert!(bad_config.validate().is_err());

        let mut bad_config = Config::default();
        bad_config.causal_min_confidence = 1.5;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_indexed_metadata_config() {
        // Default should have no indexed fields
        let config = Config::default();
        assert!(config.indexed_metadata.is_empty());

        // Test with_indexed_metadata
        let config = Config::new()
            .with_indexed_metadata(vec!["type".to_string(), "category".to_string()]);
        assert_eq!(config.indexed_metadata.len(), 2);
        assert!(config.indexed_metadata.contains(&"type".to_string()));
        assert!(config.indexed_metadata.contains(&"category".to_string()));

        // Test add_indexed_field
        let config = Config::new()
            .add_indexed_field("type")
            .add_indexed_field("priority");
        assert_eq!(config.indexed_metadata.len(), 2);
        assert!(config.indexed_metadata.contains(&"type".to_string()));
        assert!(config.indexed_metadata.contains(&"priority".to_string()));
    }
}
