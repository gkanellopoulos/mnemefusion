//! Configuration types for MnemeFusion
//!
//! This module defines configuration options for the memory engine.

use crate::query::FusionStrategy;

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
    ///
    /// **Language Note**: Entity extraction currently uses English-only stop words
    /// and capitalization rules. For non-English content, consider disabling this
    /// feature and using your own NER pipeline or relying on semantic search.
    ///
    /// See documentation for multilingual usage examples.
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

    /// Minimum semantic similarity threshold for fusion results (0.0 to 1.0)
    ///
    /// Memories with semantic_score below this threshold are excluded from 4D fusion results.
    /// This ensures that semantic relevance is mandatory - other dimensions (temporal, entity,
    /// causal) can only boost already-relevant memories, not surface irrelevant ones.
    ///
    /// Default: 0.15 (15% minimum semantic relevance)
    /// Recommended range: 0.10 to 0.20
    ///
    /// Lower values (e.g., 0.05): More permissive, may include weakly relevant memories
    /// Higher values (e.g., 0.30): Stricter, only strongly relevant memories
    ///
    /// Set to 0.0 to disable the filter (not recommended for production)
    pub fusion_semantic_threshold: f32,

    /// Pre-fusion semantic filtering threshold (0.0 to 1.0)
    ///
    /// Semantic search results below this threshold are filtered OUT before fusion.
    /// This is stricter than fusion_semantic_threshold and reduces noise in the semantic pathway.
    ///
    /// Default: 0.3 (30% minimum cosine similarity)
    /// Recommended range: 0.20 to 0.40
    ///
    /// This helps improve precision by removing low-quality semantic matches early.
    /// Set to 0.0 to disable pre-fusion filtering (uses only fusion_semantic_threshold).
    pub semantic_prefilter_threshold: f32,

    /// Fusion strategy (Weighted or ReciprocalRank)
    ///
    /// - Weighted: Uses intent-adaptive weights (original approach)
    /// - ReciprocalRank: Uses RRF formula (Hindsight's approach, proven to work better)
    ///
    /// Default: ReciprocalRank
    pub fusion_strategy: FusionStrategy,

    /// RRF k parameter (only used when fusion_strategy is ReciprocalRank)
    ///
    /// Default: 60 (from Cormack et al. 2009 RRF paper)
    pub rrf_k: f32,

    /// SLM configuration (optional)
    ///
    /// When enabled, uses Small Language Model for semantic intent classification
    /// instead of pattern matching. Improves classification accuracy from ~35% to 85%+.
    ///
    /// Default: None (disabled)
    ///
    /// Requires `slm` feature to be enabled at compile time:
    /// ```toml
    /// mnemefusion-core = { version = "0.1", features = ["slm"] }
    /// ```
    pub slm_config: Option<crate::slm::SlmConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding_dim: 384,          // Default for all-MiniLM-L6-v2
            temporal_decay_hours: 168.0, // 1 week
            causal_max_hops: 3,
            entity_extraction_enabled: true,
            causal_min_confidence: 0.5,
            hnsw_m: 16,
            hnsw_ef_construction: 128,
            hnsw_ef_search: 64,
            indexed_metadata: Vec::new(), // No indexed fields by default
            fusion_semantic_threshold: 0.15, // 15% minimum semantic relevance
            semantic_prefilter_threshold: 0.3, // 30% pre-fusion filter (Sprint 18)
            fusion_strategy: FusionStrategy::default(), // RRF by default
            rrf_k: 60.0, // From RRF paper
            slm_config: None, // SLM disabled by default
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
    ///
    /// **Language Note**: Entity extraction currently supports English only.
    /// If you're working with non-English content, set this to `false` and
    /// use multilingual embeddings for semantic search.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::Config;
    ///
    /// // For English content (default)
    /// let config = Config::new().with_entity_extraction(true);
    ///
    /// // For non-English content
    /// let config = Config::new().with_entity_extraction(false);
    /// ```
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

    /// Set the minimum semantic similarity threshold for fusion results
    ///
    /// Memories with semantic_score below this threshold are excluded from 4D fusion results.
    /// This ensures that semantic relevance is mandatory.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum semantic score (0.0 to 1.0). Default: 0.15
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::Config;
    ///
    /// // Strict filter (only highly relevant memories)
    /// let config = Config::default().with_fusion_semantic_threshold(0.30);
    ///
    /// // Permissive filter (allow weakly relevant memories)
    /// let config = Config::default().with_fusion_semantic_threshold(0.05);
    ///
    /// // Disable filter (not recommended for production)
    /// let config = Config::default().with_fusion_semantic_threshold(0.0);
    /// ```
    pub fn with_fusion_semantic_threshold(mut self, threshold: f32) -> Self {
        self.fusion_semantic_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the fusion strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - Either Weighted or ReciprocalRank
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::{Config, query::FusionStrategy};
    ///
    /// // Use RRF (default, recommended)
    /// let config = Config::default().with_fusion_strategy(FusionStrategy::ReciprocalRank);
    ///
    /// // Use weighted fusion (original approach)
    /// let config = Config::default().with_fusion_strategy(FusionStrategy::Weighted);
    /// ```
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Set the RRF k parameter
    ///
    /// Only used when fusion_strategy is ReciprocalRank.
    ///
    /// # Arguments
    ///
    /// * `k` - RRF constant (typically 60). Default: 60
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::Config;
    ///
    /// // Default k=60 (from RRF paper)
    /// let config = Config::default();
    ///
    /// // Custom k value
    /// let config = Config::default().with_rrf_k(100.0);
    /// ```
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k.max(1.0);
        self
    }

    /// Set the pre-fusion semantic filter threshold
    ///
    /// Semantic search results below this threshold are filtered out before fusion.
    /// This is stricter than fusion_semantic_threshold and helps improve precision.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum cosine similarity (0.0 to 1.0). Default: 0.3
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::Config;
    ///
    /// // Strict pre-filter (only high-quality semantic matches)
    /// let config = Config::default().with_semantic_prefilter_threshold(0.4);
    ///
    /// // Permissive pre-filter
    /// let config = Config::default().with_semantic_prefilter_threshold(0.2);
    ///
    /// // Disable pre-filter (use only fusion threshold)
    /// let config = Config::default().with_semantic_prefilter_threshold(0.0);
    /// ```
    pub fn with_semantic_prefilter_threshold(mut self, threshold: f32) -> Self {
        self.semantic_prefilter_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable SLM-based intent classification
    ///
    /// When enabled, uses Small Language Model (Gemma 3 1B) for semantic understanding
    /// of query intent, improving classification accuracy from ~35% to 85%+.
    ///
    /// Falls back to pattern-based classification on any error, ensuring zero regression.
    ///
    /// # Arguments
    ///
    /// * `slm_config` - SLM configuration including model ID and cache directory
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use mnemefusion_core::{Config, SlmConfig};
    ///
    /// // Enable SLM with default configuration
    /// let config = Config::default()
    ///     .with_slm(SlmConfig::default());
    ///
    /// // Enable SLM with custom model
    /// let config = Config::default()
    ///     .with_slm(SlmConfig::new("google/gemma-3-1b")
    ///         .with_timeout_ms(100)
    ///         .with_min_confidence(0.6));
    /// ```
    ///
    /// # Feature Flag
    ///
    /// Requires `slm` feature to be enabled:
    /// ```toml
    /// [dependencies]
    /// mnemefusion-core = { version = "0.1", features = ["slm"] }
    /// ```
    pub fn with_slm(mut self, slm_config: crate::slm::SlmConfig) -> Self {
        self.slm_config = Some(slm_config);
        self
    }

    /// Disable SLM classification (use pattern-based only)
    ///
    /// This is the default behavior. Use this method to explicitly disable SLM
    /// after it has been enabled.
    pub fn without_slm(mut self) -> Self {
        self.slm_config = None;
        self
    }

    /// Validate the configuration
    ///
    /// Returns detailed errors if the configuration is invalid.
    ///
    /// **Note**: Also prints warnings to stderr for suboptimal configurations
    /// (e.g., entity extraction enabled for potentially non-English content).
    pub fn validate(&self) -> Result<(), crate::Error> {
        // Print warning for entity extraction (English-only feature)
        if self.entity_extraction_enabled {
            eprintln!("Warning: Entity extraction is enabled. This feature currently supports English only.");
            eprintln!("         For non-English content, consider disabling with .with_entity_extraction(false)");
            eprintln!("         See documentation for multilingual usage: https://github.com/gkanellopoulos/mnemefusion");
        }

        if self.embedding_dim == 0 {
            return Err(crate::Error::Configuration(
                "embedding_dim must be greater than 0. Common values: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)".to_string(),
            ));
        }

        // Warn if dimension is unusually large
        if self.embedding_dim > 4096 {
            return Err(crate::Error::Configuration(
                format!("embedding_dim of {} is unusually large. This will consume significant memory. Typical values are 384-1536.", self.embedding_dim)
            ));
        }

        if self.temporal_decay_hours <= 0.0 {
            return Err(crate::Error::Configuration(
                "temporal_decay_hours must be positive. Recommended: 168.0 (1 week)".to_string(),
            ));
        }

        if self.causal_max_hops == 0 {
            return Err(crate::Error::Configuration(
                "causal_max_hops must be greater than 0. Recommended: 2-5".to_string(),
            ));
        }

        // Warn if hops is very large
        if self.causal_max_hops > 10 {
            return Err(crate::Error::Configuration(format!(
                "causal_max_hops of {} may be too large and cause slow queries. Recommended: 2-5",
                self.causal_max_hops
            )));
        }

        if self.causal_min_confidence < 0.0 || self.causal_min_confidence > 1.0 {
            return Err(crate::Error::Configuration(format!(
                "causal_min_confidence must be between 0.0 and 1.0, got {}",
                self.causal_min_confidence
            )));
        }

        if self.hnsw_m == 0 {
            return Err(crate::Error::Configuration(
                "hnsw_m must be greater than 0. Recommended: 12-48 (default: 16)".to_string(),
            ));
        }

        if self.hnsw_m > 100 {
            return Err(crate::Error::Configuration(format!(
                "hnsw_m of {} is very large and will consume excessive memory. Recommended: 12-48",
                self.hnsw_m
            )));
        }

        if self.hnsw_ef_construction < 10 {
            return Err(crate::Error::Configuration(
                "hnsw_ef_construction should be at least 10 for reasonable index quality. Recommended: 100-500".to_string(),
            ));
        }

        if self.hnsw_ef_search == 0 {
            return Err(crate::Error::Configuration(
                "hnsw_ef_search must be greater than 0. Recommended: 64-200".to_string(),
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
        let config =
            Config::new().with_indexed_metadata(vec!["type".to_string(), "category".to_string()]);
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

    #[test]
    fn test_config_validation_dimension_too_large() {
        let mut config = Config::default();
        config.embedding_dim = 5000;
        let err = config.validate().unwrap_err();
        assert!(matches!(err, crate::Error::Configuration(_)));
        assert!(err.to_string().contains("unusually large"));
    }

    #[test]
    fn test_config_validation_causal_hops_too_large() {
        let mut config = Config::default();
        config.causal_max_hops = 20;
        let err = config.validate().unwrap_err();
        assert!(matches!(err, crate::Error::Configuration(_)));
        assert!(err.to_string().contains("too large"));
    }

    #[test]
    fn test_config_validation_hnsw_m_too_large() {
        let mut config = Config::default();
        config.hnsw_m = 150;
        let err = config.validate().unwrap_err();
        assert!(matches!(err, crate::Error::Configuration(_)));
        assert!(err.to_string().contains("very large"));
    }

    #[test]
    fn test_config_validation_ef_construction_too_small() {
        let mut config = Config::default();
        config.hnsw_ef_construction = 5;
        let err = config.validate().unwrap_err();
        assert!(matches!(err, crate::Error::Configuration(_)));
        assert!(err.to_string().contains("at least 10"));
    }

    #[test]
    fn test_config_validation_ef_search_zero() {
        let mut config = Config::default();
        config.hnsw_ef_search = 0;
        let err = config.validate().unwrap_err();
        assert!(matches!(err, crate::Error::Configuration(_)));
    }

    #[test]
    fn test_config_validation_provides_recommendations() {
        // Test that error messages include recommendations
        let mut config = Config::default();
        config.embedding_dim = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("Common values"));

        let mut config = Config::default();
        config.temporal_decay_hours = 0.0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("Recommended"));
    }

    #[test]
    fn test_entity_extraction_warning() {
        // Entity extraction enabled should validate successfully but print warning
        let config = Config::default();
        assert!(config.entity_extraction_enabled); // Default is true

        // Should not error, just warn to stderr
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_entity_extraction_disabled_no_warning() {
        // Disabling entity extraction should validate without warnings
        let config = Config::new().with_entity_extraction(false);
        assert!(!config.entity_extraction_enabled);

        // Should validate successfully
        assert!(config.validate().is_ok());
    }
}
