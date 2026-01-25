//! Result fusion engine for combining multi-dimensional search results
//!
//! Combines semantic, temporal, causal, and entity search results with
//! adaptive weighting based on query intent.

use crate::{query::intent::QueryIntent, types::MemoryId};
use std::collections::HashMap;

/// Configuration for adaptive weight selection based on intent
#[derive(Debug, Clone)]
pub struct AdaptiveWeightConfig {
    /// Weights for temporal queries
    pub temporal: IntentWeights,
    /// Weights for causal queries
    pub causal: IntentWeights,
    /// Weights for entity queries
    pub entity: IntentWeights,
    /// Weights for factual queries
    pub factual: IntentWeights,
}

/// Weights for each dimension
#[derive(Debug, Clone, Copy)]
pub struct IntentWeights {
    /// Weight for semantic similarity (0.0 to 1.0)
    pub semantic: f32,
    /// Weight for temporal relevance (0.0 to 1.0)
    pub temporal: f32,
    /// Weight for causal relevance (0.0 to 1.0)
    pub causal: f32,
    /// Weight for entity relevance (0.0 to 1.0)
    pub entity: f32,
}

impl IntentWeights {
    /// Create weights that sum to 1.0
    pub fn new(semantic: f32, temporal: f32, causal: f32, entity: f32) -> Self {
        let total = semantic + temporal + causal + entity;
        Self {
            semantic: semantic / total,
            temporal: temporal / total,
            causal: causal / total,
            entity: entity / total,
        }
    }

    /// Validate that weights sum to approximately 1.0
    pub fn validate(&self) -> bool {
        let sum = self.semantic + self.temporal + self.causal + self.entity;
        (sum - 1.0).abs() < 0.01
    }
}

impl Default for AdaptiveWeightConfig {
    fn default() -> Self {
        Self {
            // Temporal queries: prioritize time, then semantic
            temporal: IntentWeights::new(0.3, 0.5, 0.1, 0.1),
            // Causal queries: prioritize causal links, then semantic
            causal: IntentWeights::new(0.3, 0.1, 0.5, 0.1),
            // Entity queries: prioritize entity links, then semantic
            entity: IntentWeights::new(0.3, 0.1, 0.1, 0.5),
            // Factual queries: pure semantic search
            factual: IntentWeights::new(0.8, 0.1, 0.05, 0.05),
        }
    }
}

/// A fused search result with scores from multiple dimensions
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// The memory ID
    pub id: MemoryId,
    /// Semantic similarity score (0.0 to 1.0)
    pub semantic_score: f32,
    /// Temporal relevance score (0.0 to 1.0)
    pub temporal_score: f32,
    /// Causal relevance score (0.0 to 1.0)
    pub causal_score: f32,
    /// Entity relevance score (0.0 to 1.0)
    pub entity_score: f32,
    /// Final fused score (0.0 to 1.0)
    pub fused_score: f32,
}

/// Fusion engine for combining multi-dimensional search results
pub struct FusionEngine {
    config: AdaptiveWeightConfig,
}

impl FusionEngine {
    /// Create a new fusion engine with default weights
    pub fn new() -> Self {
        Self {
            config: AdaptiveWeightConfig::default(),
        }
    }

    /// Create a fusion engine with custom weight configuration
    pub fn with_config(config: AdaptiveWeightConfig) -> Self {
        Self { config }
    }

    /// Get weights for a specific intent
    pub fn get_weights(&self, intent: QueryIntent) -> IntentWeights {
        match intent {
            QueryIntent::Temporal => self.config.temporal,
            QueryIntent::Causal => self.config.causal,
            QueryIntent::Entity => self.config.entity,
            QueryIntent::Factual => self.config.factual,
        }
    }

    /// Fuse results from multiple dimensions
    ///
    /// Takes results from each dimension and combines them using adaptive weights
    /// based on the query intent.
    ///
    /// # Arguments
    ///
    /// * `intent` - The classified query intent
    /// * `semantic_results` - Results from semantic search (memory_id -> score)
    /// * `temporal_results` - Results from temporal search (memory_id -> score)
    /// * `causal_results` - Results from causal search (memory_id -> score)
    /// * `entity_results` - Results from entity search (memory_id -> score)
    ///
    /// # Returns
    ///
    /// Vector of FusedResult sorted by fused_score descending
    pub fn fuse(
        &self,
        intent: QueryIntent,
        semantic_results: &HashMap<MemoryId, f32>,
        temporal_results: &HashMap<MemoryId, f32>,
        causal_results: &HashMap<MemoryId, f32>,
        entity_results: &HashMap<MemoryId, f32>,
    ) -> Vec<FusedResult> {
        let weights = self.get_weights(intent);

        // Collect all unique memory IDs
        let mut all_ids = std::collections::HashSet::new();
        all_ids.extend(semantic_results.keys());
        all_ids.extend(temporal_results.keys());
        all_ids.extend(causal_results.keys());
        all_ids.extend(entity_results.keys());

        // Compute fused scores
        let mut results: Vec<FusedResult> = all_ids
            .into_iter()
            .map(|id| {
                let semantic_score = *semantic_results.get(id).unwrap_or(&0.0);
                let temporal_score = *temporal_results.get(id).unwrap_or(&0.0);
                let causal_score = *causal_results.get(id).unwrap_or(&0.0);
                let entity_score = *entity_results.get(id).unwrap_or(&0.0);

                // Compute weighted sum
                let fused_score = semantic_score * weights.semantic
                    + temporal_score * weights.temporal
                    + causal_score * weights.causal
                    + entity_score * weights.entity;

                FusedResult {
                    id: id.clone(),
                    semantic_score,
                    temporal_score,
                    causal_score,
                    entity_score,
                    fused_score,
                }
            })
            .collect();

        // Sort by fused score descending
        results.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap());

        results
    }

    /// Normalize scores to 0.0-1.0 range
    ///
    /// Useful for raw scores that may be in different ranges.
    pub fn normalize_scores(scores: &mut HashMap<MemoryId, f32>) {
        if scores.is_empty() {
            return;
        }

        let max_score = scores.values().copied().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory_id(n: u128) -> MemoryId {
        MemoryId::from_u128(n)
    }

    #[test]
    fn test_intent_weights_normalization() {
        let weights = IntentWeights::new(1.0, 2.0, 3.0, 4.0);
        assert!(weights.validate());
        assert!(
            (weights.semantic + weights.temporal + weights.causal + weights.entity - 1.0).abs()
                < 0.01
        );
    }

    #[test]
    fn test_default_weights() {
        let config = AdaptiveWeightConfig::default();
        assert!(config.temporal.validate());
        assert!(config.causal.validate());
        assert!(config.entity.validate());
        assert!(config.factual.validate());
    }

    #[test]
    fn test_fusion_temporal_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Temporal);

        // Temporal queries should weight temporal dimension highest
        assert!(weights.temporal > weights.semantic);
        assert!(weights.temporal > weights.causal);
        assert!(weights.temporal > weights.entity);
    }

    #[test]
    fn test_fusion_causal_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Causal);

        // Causal queries should weight causal dimension highest
        assert!(weights.causal > weights.semantic);
        assert!(weights.causal > weights.temporal);
        assert!(weights.causal > weights.entity);
    }

    #[test]
    fn test_fusion_entity_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Entity);

        // Entity queries should weight entity dimension highest
        assert!(weights.entity > weights.semantic);
        assert!(weights.entity > weights.temporal);
        assert!(weights.entity > weights.causal);
    }

    #[test]
    fn test_fusion_factual_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Factual);

        // Factual queries should weight semantic dimension highest
        assert!(weights.semantic > weights.temporal);
        assert!(weights.semantic > weights.causal);
        assert!(weights.semantic > weights.entity);
    }

    #[test]
    fn test_fuse_single_dimension() {
        let engine = FusionEngine::new();

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 1.0);
        semantic.insert(make_memory_id(2), 0.5);

        let temporal = HashMap::new();
        let causal = HashMap::new();
        let entity = HashMap::new();

        let results = engine.fuse(QueryIntent::Factual, &semantic, &temporal, &causal, &entity);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, make_memory_id(1));
        assert_eq!(results[1].id, make_memory_id(2));
        assert!(results[0].fused_score > results[1].fused_score);
    }

    #[test]
    fn test_fuse_multiple_dimensions() {
        let engine = FusionEngine::new();

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.5);
        semantic.insert(make_memory_id(2), 0.3);

        let mut temporal = HashMap::new();
        temporal.insert(make_memory_id(1), 1.0);
        temporal.insert(make_memory_id(3), 0.8);

        let causal = HashMap::new();
        let entity = HashMap::new();

        // Use temporal intent (weights temporal highly)
        let results = engine.fuse(
            QueryIntent::Temporal,
            &semantic,
            &temporal,
            &causal,
            &entity,
        );

        assert_eq!(results.len(), 3);
        // ID 1 should rank highest (has both semantic and temporal)
        assert_eq!(results[0].id, make_memory_id(1));
    }

    #[test]
    fn test_normalize_scores() {
        let mut scores = HashMap::new();
        scores.insert(make_memory_id(1), 100.0);
        scores.insert(make_memory_id(2), 50.0);
        scores.insert(make_memory_id(3), 25.0);

        FusionEngine::normalize_scores(&mut scores);

        assert!((scores[&make_memory_id(1)] - 1.0).abs() < 0.01);
        assert!((scores[&make_memory_id(2)] - 0.5).abs() < 0.01);
        assert!((scores[&make_memory_id(3)] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_fused_result_sorting() {
        let engine = FusionEngine::new();

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.3);
        semantic.insert(make_memory_id(2), 0.8);
        semantic.insert(make_memory_id(3), 0.5);

        let results = engine.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );

        // Should be sorted by fused_score descending
        assert_eq!(results[0].id, make_memory_id(2));
        assert_eq!(results[1].id, make_memory_id(3));
        assert_eq!(results[2].id, make_memory_id(1));
    }

    #[test]
    fn test_empty_results() {
        let engine = FusionEngine::new();

        let results = engine.fuse(
            QueryIntent::Factual,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );

        assert_eq!(results.len(), 0);
    }
}
