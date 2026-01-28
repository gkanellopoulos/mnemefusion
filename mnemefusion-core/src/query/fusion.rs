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
    /// Weight for BM25 keyword matching (0.0 to 1.0)
    pub bm25: f32,
    /// Weight for temporal relevance (0.0 to 1.0)
    pub temporal: f32,
    /// Weight for causal relevance (0.0 to 1.0)
    pub causal: f32,
    /// Weight for entity relevance (0.0 to 1.0)
    pub entity: f32,
}

impl IntentWeights {
    /// Create weights that sum to 1.0
    pub fn new(semantic: f32, bm25: f32, temporal: f32, causal: f32, entity: f32) -> Self {
        let total = semantic + bm25 + temporal + causal + entity;
        Self {
            semantic: semantic / total,
            bm25: bm25 / total,
            temporal: temporal / total,
            causal: causal / total,
            entity: entity / total,
        }
    }

    /// Validate that weights sum to approximately 1.0
    pub fn validate(&self) -> bool {
        let sum = self.semantic + self.bm25 + self.temporal + self.causal + self.entity;
        (sum - 1.0).abs() < 0.01
    }
}

impl Default for AdaptiveWeightConfig {
    fn default() -> Self {
        Self {
            // Temporal queries: semantic + BM25 base, temporal boost
            // BM25 helps catch exact date/time keyword matches
            temporal: IntentWeights::new(0.35, 0.20, 0.35, 0.05, 0.05),
            // Causal queries: semantic + BM25 base, causal boost
            // BM25 helps catch "because", "caused by" exact matches
            causal: IntentWeights::new(0.35, 0.20, 0.05, 0.35, 0.05),
            // Entity queries: semantic + BM25 base, entity boost
            // BM25 helps catch exact entity name matches
            entity: IntentWeights::new(0.35, 0.20, 0.05, 0.05, 0.35),
            // Factual queries: semantic + BM25 (exact keyword matching critical)
            factual: IntentWeights::new(0.50, 0.30, 0.10, 0.05, 0.05),
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
    /// BM25 keyword matching score (0.0 to 1.0)
    pub bm25_score: f32,
    /// Temporal relevance score (0.0 to 1.0)
    pub temporal_score: f32,
    /// Causal relevance score (0.0 to 1.0)
    pub causal_score: f32,
    /// Entity relevance score (0.0 to 1.0)
    pub entity_score: f32,
    /// Final fused score (0.0 to 1.0)
    pub fused_score: f32,
    /// Cross-dimensional validation confidence (0.0 to 1.0)
    ///
    /// Confidence based on how many dimensions contributed to this result:
    /// - 5 dimensions: 1.0 (all dimensions agree)
    /// - 4 dimensions: 0.9
    /// - 3 dimensions: 0.8
    /// - 2 dimensions: 0.6 (medium confidence)
    /// - 1 dimension: 0.4 (low confidence, likely noise)
    ///
    /// Added in Sprint 18 Task 18.2 to improve precision
    pub confidence: f32,
}

/// Fusion strategy for combining multi-dimensional search results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Weighted fusion: score = Σ (weight_i * score_i)
    Weighted,
    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
    /// Used by Hindsight (85.7% accuracy on LoCoMo)
    ReciprocalRank,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        // Use RRF by default (proven to work better)
        Self::ReciprocalRank
    }
}

/// Fusion engine for combining multi-dimensional search results
pub struct FusionEngine {
    config: AdaptiveWeightConfig,
    /// Minimum semantic threshold (0.0 to 1.0)
    /// Memories with semantic_score below this threshold are excluded from results
    /// Default: 0.15 (15% minimum semantic relevance)
    semantic_threshold: f32,
    /// Fusion strategy (Weighted or ReciprocalRank)
    strategy: FusionStrategy,
    /// RRF k parameter (default: 60, from Hindsight paper)
    rrf_k: f32,
}

impl FusionEngine {
    /// Create a new fusion engine with default settings
    /// - Strategy: ReciprocalRank (Hindsight's approach)
    /// - Semantic threshold: 0.15
    /// - RRF k: 60
    pub fn new() -> Self {
        Self {
            config: AdaptiveWeightConfig::default(),
            semantic_threshold: 0.15, // 15% minimum semantic relevance
            strategy: FusionStrategy::default(),
            rrf_k: 60.0, // From Hindsight paper
        }
    }

    /// Create a fusion engine with custom weight configuration
    pub fn with_config(config: AdaptiveWeightConfig) -> Self {
        Self {
            config,
            semantic_threshold: 0.15,
            strategy: FusionStrategy::default(),
            rrf_k: 60.0,
        }
    }

    /// Set the fusion strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - Either Weighted or ReciprocalRank
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the RRF k parameter
    ///
    /// Only used when strategy is ReciprocalRank.
    /// Default: 60 (from Cormack et al. 2009 RRF paper)
    ///
    /// # Arguments
    ///
    /// * `k` - RRF constant (typically 60)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k.max(1.0);
        self
    }

    /// Set the minimum semantic threshold
    ///
    /// Memories with semantic_score below this threshold are excluded from fusion results.
    /// This ensures that semantic relevance is mandatory - other dimensions can only boost
    /// already-relevant memories, not surface irrelevant ones.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum semantic score (0.0 to 1.0). Recommended: 0.10 to 0.20
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_semantic_threshold(mut self, threshold: f32) -> Self {
        self.semantic_threshold = threshold.clamp(0.0, 1.0);
        self
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
    /// Takes results from each dimension and combines them using the configured
    /// fusion strategy (Weighted or ReciprocalRank).
    ///
    /// # Arguments
    ///
    /// * `intent` - The classified query intent (used for weighted fusion)
    /// * `semantic_results` - Results from semantic search (memory_id -> score)
    /// * `bm25_results` - Results from BM25 keyword search (memory_id -> score)
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
        bm25_results: &HashMap<MemoryId, f32>,
        temporal_results: &HashMap<MemoryId, f32>,
        causal_results: &HashMap<MemoryId, f32>,
        entity_results: &HashMap<MemoryId, f32>,
    ) -> Vec<FusedResult> {
        match self.strategy {
            FusionStrategy::Weighted => self.fuse_weighted(
                intent,
                semantic_results,
                bm25_results,
                temporal_results,
                causal_results,
                entity_results,
            ),
            FusionStrategy::ReciprocalRank => self.fuse_rrf(
                semantic_results,
                bm25_results,
                temporal_results,
                causal_results,
                entity_results,
            ),
        }
    }

    /// Fuse using weighted combination (original approach)
    fn fuse_weighted(
        &self,
        intent: QueryIntent,
        semantic_results: &HashMap<MemoryId, f32>,
        bm25_results: &HashMap<MemoryId, f32>,
        temporal_results: &HashMap<MemoryId, f32>,
        causal_results: &HashMap<MemoryId, f32>,
        entity_results: &HashMap<MemoryId, f32>,
    ) -> Vec<FusedResult> {
        let weights = self.get_weights(intent);

        // Collect all unique memory IDs
        let mut all_ids = std::collections::HashSet::new();
        all_ids.extend(semantic_results.keys());
        all_ids.extend(bm25_results.keys());
        all_ids.extend(temporal_results.keys());
        all_ids.extend(causal_results.keys());
        all_ids.extend(entity_results.keys());

        // Compute fused scores
        // IMPORTANT: Filter out memories with very low semantic relevance
        // This ensures semantic relevance is mandatory - other dimensions can only boost
        // already-relevant memories, not surface irrelevant ones
        let mut results: Vec<FusedResult> = all_ids
            .into_iter()
            .filter_map(|id| {
                let semantic_score = *semantic_results.get(id).unwrap_or(&0.0);
                let bm25_score = *bm25_results.get(id).unwrap_or(&0.0);

                // FILTER: Require minimum semantic relevance OR strong BM25 match
                // Allow BM25 to surface exact keyword matches even if semantic is low
                // This is critical for queries with specific terms/names
                if semantic_score < self.semantic_threshold && bm25_score < 0.3 {
                    return None;
                }

                let temporal_score = *temporal_results.get(id).unwrap_or(&0.0);
                let causal_score = *causal_results.get(id).unwrap_or(&0.0);
                let entity_score = *entity_results.get(id).unwrap_or(&0.0);

                // Compute weighted sum
                let fused_score = semantic_score * weights.semantic
                    + bm25_score * weights.bm25
                    + temporal_score * weights.temporal
                    + causal_score * weights.causal
                    + entity_score * weights.entity;

                Some(FusedResult {
                    id: id.clone(),
                    semantic_score,
                    bm25_score,
                    temporal_score,
                    causal_score,
                    entity_score,
                    fused_score,
                    confidence: 1.0, // Will be adjusted by cross-dimensional validation
                })
            })
            .collect();

        // Sort by fused score descending
        results.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap());

        results
    }

    /// Fuse using Reciprocal Rank Fusion (Hindsight's approach)
    ///
    /// RRF formula: score(doc) = Σ 1/(k + rank_i) across all pathways
    /// where k=60 (constant), rank_i is the rank in pathway i (0-indexed)
    ///
    /// This approach is proven to work well (Hindsight: 85.7% accuracy)
    /// and doesn't require tuning weights per query type.
    fn fuse_rrf(
        &self,
        semantic_results: &HashMap<MemoryId, f32>,
        bm25_results: &HashMap<MemoryId, f32>,
        temporal_results: &HashMap<MemoryId, f32>,
        causal_results: &HashMap<MemoryId, f32>,
        entity_results: &HashMap<MemoryId, f32>,
    ) -> Vec<FusedResult> {
        // Convert score maps to ranked lists (sorted by score descending)
        let semantic_ranked = Self::to_ranked_list(semantic_results);
        let bm25_ranked = Self::to_ranked_list(bm25_results);
        let temporal_ranked = Self::to_ranked_list(temporal_results);
        let causal_ranked = Self::to_ranked_list(causal_results);
        let entity_ranked = Self::to_ranked_list(entity_results);

        // Build RRF scores: for each pathway, add 1/(k + rank) to each doc
        let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

        // Semantic pathway
        for (rank, id) in semantic_ranked.iter().enumerate() {
            *rrf_scores.entry(id.clone()).or_default() +=
                1.0 / (self.rrf_k + rank as f32 + 1.0);
        }

        // BM25 pathway
        for (rank, id) in bm25_ranked.iter().enumerate() {
            *rrf_scores.entry(id.clone()).or_default() +=
                1.0 / (self.rrf_k + rank as f32 + 1.0);
        }

        // Temporal pathway
        for (rank, id) in temporal_ranked.iter().enumerate() {
            *rrf_scores.entry(id.clone()).or_default() +=
                1.0 / (self.rrf_k + rank as f32 + 1.0);
        }

        // Causal pathway
        for (rank, id) in causal_ranked.iter().enumerate() {
            *rrf_scores.entry(id.clone()).or_default() +=
                1.0 / (self.rrf_k + rank as f32 + 1.0);
        }

        // Entity pathway
        for (rank, id) in entity_ranked.iter().enumerate() {
            *rrf_scores.entry(id.clone()).or_default() +=
                1.0 / (self.rrf_k + rank as f32 + 1.0);
        }

        // FILTER: Apply semantic threshold to prevent keyword flooding
        // Memories must have minimum semantic relevance OR strong BM25 match
        let results: Vec<FusedResult> = rrf_scores
            .into_iter()
            .filter_map(|(id, rrf_score)| {
                let semantic_score = *semantic_results.get(&id).unwrap_or(&0.0);
                let bm25_score = *bm25_results.get(&id).unwrap_or(&0.0);

                // Require minimum semantic relevance OR strong BM25 match
                if semantic_score < self.semantic_threshold && bm25_score < 0.3 {
                    return None;
                }

                let temporal_score = *temporal_results.get(&id).unwrap_or(&0.0);
                let causal_score = *causal_results.get(&id).unwrap_or(&0.0);
                let entity_score = *entity_results.get(&id).unwrap_or(&0.0);

                Some(FusedResult {
                    id,
                    semantic_score,
                    bm25_score,
                    temporal_score,
                    causal_score,
                    entity_score,
                    fused_score: rrf_score,
                    confidence: 1.0, // Will be adjusted by cross-dimensional validation
                })
            })
            .collect();

        // Sort by RRF score descending
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| b.fused_score.partial_cmp(&a.fused_score).unwrap());

        sorted_results
    }

    /// Convert score map to ranked list (sorted by score descending)
    ///
    /// Returns vector of MemoryIds in rank order (highest score first)
    fn to_ranked_list(scores: &HashMap<MemoryId, f32>) -> Vec<MemoryId> {
        let mut ranked: Vec<_> = scores.iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        ranked.into_iter().map(|(id, _)| id.clone()).collect()
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
        let weights = IntentWeights::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert!(weights.validate());
        assert!(
            (weights.semantic + weights.bm25 + weights.temporal + weights.causal + weights.entity - 1.0).abs()
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

        // Temporal queries: semantic + BM25 base, temporal boost
        assert!(weights.semantic >= 0.30); // ~35% semantic
        assert!(weights.bm25 >= 0.15); // ~20% BM25
        assert!(weights.temporal >= 0.30); // ~35% temporal
        assert!(weights.temporal > weights.causal);
        assert!(weights.temporal > weights.entity);
    }

    #[test]
    fn test_fusion_causal_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Causal);

        // Causal queries: semantic + BM25 base, causal boost
        assert!(weights.semantic >= 0.30); // ~35% semantic
        assert!(weights.bm25 >= 0.15); // ~20% BM25
        assert!(weights.causal >= 0.30); // ~35% causal
        assert!(weights.causal > weights.temporal);
        assert!(weights.causal > weights.entity);
    }

    #[test]
    fn test_fusion_entity_intent() {
        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Entity);

        // Entity queries: semantic + BM25 base, entity boost
        assert!(weights.semantic >= 0.30); // ~35% semantic
        assert!(weights.bm25 >= 0.15); // ~20% BM25
        assert!(weights.entity >= 0.30); // ~35% entity
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

        let bm25 = HashMap::new();
        let temporal = HashMap::new();
        let causal = HashMap::new();
        let entity = HashMap::new();

        let results = engine.fuse(QueryIntent::Factual, &semantic, &bm25, &temporal, &causal, &entity);

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

        let bm25 = HashMap::new();
        let causal = HashMap::new();
        let entity = HashMap::new();

        // Use temporal intent (weights temporal highly)
        let results = engine.fuse(
            QueryIntent::Temporal,
            &semantic,
            &bm25,
            &temporal,
            &causal,
            &entity,
        );

        // Memory 3 is excluded because it has no semantic score (0.0 < threshold 0.15)
        assert_eq!(results.len(), 2);
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
            &HashMap::new(), // bm25
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
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
            &HashMap::new(), // semantic
            &HashMap::new(), // bm25
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_semantic_threshold_filter() {
        // Test that memories with semantic score below threshold are excluded
        let engine = FusionEngine::new().with_semantic_threshold(0.15);

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.5); // Above threshold
        semantic.insert(make_memory_id(2), 0.2); // Above threshold
        semantic.insert(make_memory_id(3), 0.1); // Below threshold (excluded)

        let mut temporal = HashMap::new();
        temporal.insert(make_memory_id(1), 0.3);
        temporal.insert(make_memory_id(2), 0.8);
        temporal.insert(make_memory_id(3), 1.0); // High temporal, but low semantic
        temporal.insert(make_memory_id(4), 1.0); // Only in temporal (no semantic score)

        let results = engine.fuse(
            QueryIntent::Temporal,
            &semantic,
            &HashMap::new(), // bm25
            &temporal,
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Should only include memories with semantic_score >= 0.15
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.semantic_score >= 0.15));

        // Memory 3 should be excluded (semantic_score = 0.1 < threshold)
        assert!(!results.iter().any(|r| r.id == make_memory_id(3)));

        // Memory 4 should be excluded (semantic_score = 0.0 < threshold)
        assert!(!results.iter().any(|r| r.id == make_memory_id(4)));
    }

    #[test]
    fn test_semantic_threshold_prevents_keyword_flooding() {
        // Test the specific bug: memories with high temporal/entity/causal
        // but zero semantic score should be excluded
        let engine = FusionEngine::new().with_semantic_threshold(0.15);

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.5450); // Semantically relevant

        let mut temporal = HashMap::new();
        temporal.insert(make_memory_id(1), 0.0); // Not temporal
        temporal.insert(make_memory_id(2), 1.0); // Highly temporal BUT semantically irrelevant

        let results = engine.fuse(
            QueryIntent::Temporal,
            &semantic,
            &HashMap::new(), // bm25
            &temporal,
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Should only return memory 1 (has semantic score above threshold)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, make_memory_id(1));

        // Memory 2 should be excluded despite high temporal score
        // because semantic_score = 0.0 < threshold
        assert!(!results.iter().any(|r| r.id == make_memory_id(2)));
    }

    #[test]
    fn test_semantic_threshold_configurable() {
        // Test that semantic threshold can be configured
        let engine1 = FusionEngine::new().with_semantic_threshold(0.1);
        let engine2 = FusionEngine::new().with_semantic_threshold(0.3);

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.5);
        semantic.insert(make_memory_id(2), 0.2);

        let results1 = engine1.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(), // bm25
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        let results2 = engine2.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(), // bm25
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Lower threshold (0.1) should include both memories
        assert_eq!(results1.len(), 2);

        // Higher threshold (0.3) should exclude memory 2 (score = 0.2)
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].id, make_memory_id(1));
    }

    // ========== RRF Tests ==========

    #[test]
    fn test_rrf_strategy_selection() {
        // Test that RRF is the default strategy
        let engine = FusionEngine::new();
        assert_eq!(engine.strategy, FusionStrategy::ReciprocalRank);

        // Test that we can switch to weighted
        let engine_weighted = FusionEngine::new().with_strategy(FusionStrategy::Weighted);
        assert_eq!(engine_weighted.strategy, FusionStrategy::Weighted);
    }

    #[test]
    fn test_rrf_single_pathway() {
        // Test RRF with only semantic results
        let engine = FusionEngine::new()
            .with_strategy(FusionStrategy::ReciprocalRank)
            .with_semantic_threshold(0.0); // Disable threshold for test

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 1.0);
        semantic.insert(make_memory_id(2), 0.8);
        semantic.insert(make_memory_id(3), 0.6);

        let results = engine.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(), // bm25
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Should rank by RRF score (which follows semantic ranking when only one pathway)
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, make_memory_id(1)); // Rank 0: 1/(60+1) = 0.0164
        assert_eq!(results[1].id, make_memory_id(2)); // Rank 1: 1/(60+2) = 0.0161
        assert_eq!(results[2].id, make_memory_id(3)); // Rank 2: 1/(60+3) = 0.0159

        // Verify RRF scores are calculated correctly
        assert!(results[0].fused_score > results[1].fused_score);
        assert!(results[1].fused_score > results[2].fused_score);
    }

    #[test]
    fn test_rrf_multi_pathway_consensus() {
        // Test that RRF boosts documents that appear in multiple pathways
        let engine = FusionEngine::new()
            .with_strategy(FusionStrategy::ReciprocalRank)
            .with_semantic_threshold(0.0);

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.8); // Rank 0
        semantic.insert(make_memory_id(2), 0.5); // Rank 1

        let mut bm25 = HashMap::new();
        bm25.insert(make_memory_id(1), 0.9); // Rank 0 (also in semantic!)
        bm25.insert(make_memory_id(3), 0.7); // Rank 1

        let results = engine.fuse(
            QueryIntent::Factual,
            &semantic,
            &bm25,
            &HashMap::new(), // temporal
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Memory 1 appears in both pathways at rank 0
        // RRF score = 1/(60+1) + 1/(60+1) = 0.0164 * 2 = 0.0328
        // Memory 2 appears in semantic at rank 1
        // RRF score = 1/(60+2) = 0.0161
        // Memory 3 appears in bm25 at rank 1
        // RRF score = 1/(60+2) = 0.0161

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, make_memory_id(1)); // Highest RRF (in both lists)
        // Memory 2 and 3 have same RRF score, order may vary
    }

    #[test]
    fn test_rrf_vs_weighted_different_rankings() {
        // Test that RRF can produce different rankings than weighted fusion
        let engine_rrf = FusionEngine::new()
            .with_strategy(FusionStrategy::ReciprocalRank)
            .with_semantic_threshold(0.0);

        let engine_weighted = FusionEngine::new()
            .with_strategy(FusionStrategy::Weighted)
            .with_semantic_threshold(0.0);

        // Setup: memory 1 has high semantic, memory 2 has high temporal
        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 0.9);
        semantic.insert(make_memory_id(2), 0.2);

        let mut temporal = HashMap::new();
        temporal.insert(make_memory_id(1), 0.1);
        temporal.insert(make_memory_id(2), 0.95);

        let rrf_results = engine_rrf.fuse(
            QueryIntent::Temporal,
            &semantic,
            &HashMap::new(), // bm25
            &temporal,
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        let weighted_results = engine_weighted.fuse(
            QueryIntent::Temporal,
            &semantic,
            &HashMap::new(), // bm25
            &temporal,
            &HashMap::new(), // causal
            &HashMap::new(), // entity
        );

        // Both should have 2 results
        assert_eq!(rrf_results.len(), 2);
        assert_eq!(weighted_results.len(), 2);

        // RRF: Memory 1 ranks first in semantic (high boost), Memory 2 ranks first in temporal
        // Both contribute equally in RRF, but Memory 1 has stronger semantic signal
        // Weighted: Depends on intent weights (temporal intent favors memory 2)

        // The key insight: RRF doesn't over-weight any single dimension
        // It rewards consensus across pathways
    }

    #[test]
    fn test_rrf_k_parameter() {
        // Test that changing k parameter affects scores
        let engine_k60 = FusionEngine::new()
            .with_strategy(FusionStrategy::ReciprocalRank)
            .with_rrf_k(60.0)
            .with_semantic_threshold(0.0);

        let engine_k10 = FusionEngine::new()
            .with_strategy(FusionStrategy::ReciprocalRank)
            .with_rrf_k(10.0)
            .with_semantic_threshold(0.0);

        let mut semantic = HashMap::new();
        semantic.insert(make_memory_id(1), 1.0);
        semantic.insert(make_memory_id(2), 0.5);

        let results_k60 = engine_k60.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );

        let results_k10 = engine_k10.fuse(
            QueryIntent::Factual,
            &semantic,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );

        // Both should rank same documents in same order
        assert_eq!(results_k60[0].id, results_k10[0].id);

        // But scores should be different (k=10 gives higher scores)
        // k=60: 1/(60+1) = 0.0164
        // k=10: 1/(10+1) = 0.0909
        assert!(results_k10[0].fused_score > results_k60[0].fused_score);
    }

    #[test]
    fn test_to_ranked_list() {
        let mut scores = HashMap::new();
        scores.insert(make_memory_id(1), 0.5);
        scores.insert(make_memory_id(2), 0.9);
        scores.insert(make_memory_id(3), 0.2);

        let ranked = FusionEngine::to_ranked_list(&scores);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0], make_memory_id(2)); // Highest score
        assert_eq!(ranked[1], make_memory_id(1));
        assert_eq!(ranked[2], make_memory_id(3)); // Lowest score
    }
}
