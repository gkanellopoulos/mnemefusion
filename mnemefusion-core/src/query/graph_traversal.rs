//! Policy-guided graph traversal for retrieval expansion
//!
//! This module implements intelligent graph traversal to expand retrieval results
//! beyond initial semantic matches. Different policies are applied based on query intent.

use crate::error::Result;
use crate::graph::GraphManager;
use crate::query::intent::QueryIntent;
use crate::storage::StorageEngine;
use crate::types::MemoryId;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Configuration for graph traversal
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum hops for causal traversal
    pub causal_max_hops: usize,
    /// Maximum hops for entity traversal
    pub entity_max_hops: usize,
    /// Minimum confidence for causal edges to follow
    pub causal_min_confidence: f32,
    /// Decay factor for traversal scores (0.0-1.0)
    /// Score is multiplied by this factor for each hop
    pub decay_factor: f32,
    /// Maximum number of expanded results per seed
    pub max_expansions_per_seed: usize,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            causal_max_hops: 2,
            entity_max_hops: 1,
            causal_min_confidence: 0.3,
            decay_factor: 0.7,
            max_expansions_per_seed: 5,
        }
    }
}

/// Policy-guided graph traversal engine
pub struct GraphTraversal {
    graph_manager: Arc<RwLock<GraphManager>>,
    storage: Arc<StorageEngine>,
    config: TraversalConfig,
}

impl GraphTraversal {
    /// Create a new graph traversal engine
    pub fn new(
        graph_manager: Arc<RwLock<GraphManager>>,
        storage: Arc<StorageEngine>,
        config: TraversalConfig,
    ) -> Self {
        Self {
            graph_manager,
            storage,
            config,
        }
    }

    /// Expand retrieval results using graph traversal
    ///
    /// Takes seed results (from semantic/BM25 search) and expands them using
    /// graph traversal based on query intent. Returns HashMap of expanded
    /// memories with their scores.
    ///
    /// # Arguments
    /// * `seed_results` - Initial retrieval results with scores
    /// * `intent` - Query intent to determine traversal policy
    /// * `limit` - Maximum number of total results (seeds + expansions)
    pub fn expand(
        &self,
        seed_results: &HashMap<MemoryId, f32>,
        intent: QueryIntent,
        limit: usize,
    ) -> Result<HashMap<MemoryId, f32>> {
        // Start with seed results
        let mut expanded = seed_results.clone();

        // Apply intent-specific traversal policy
        match intent {
            QueryIntent::Causal => {
                self.expand_causal(&mut expanded, seed_results, limit)?;
            }
            QueryIntent::Entity => {
                self.expand_entity(&mut expanded, seed_results, limit)?;
            }
            QueryIntent::Temporal => {
                // Temporal expansion: find causally or entity-related memories
                // that might provide temporal context
                self.expand_hybrid(&mut expanded, seed_results, limit)?;
            }
            QueryIntent::Factual => {
                // Factual queries benefit most from entity-based expansion
                self.expand_entity(&mut expanded, seed_results, limit)?;
            }
        }

        Ok(expanded)
    }

    /// Expand using causal graph traversal
    ///
    /// For each seed result, traverse causal graph (causes and effects)
    /// to find related memories that might answer the query.
    fn expand_causal(
        &self,
        expanded: &mut HashMap<MemoryId, f32>,
        seed_results: &HashMap<MemoryId, f32>,
        limit: usize,
    ) -> Result<()> {
        let graph = self.graph_manager.read().unwrap();

        // Track which seeds we've already expanded
        let mut visited = HashSet::new();

        // Sort seeds by score (highest first) for expansion priority
        let mut seed_vec: Vec<_> = seed_results.iter().collect();
        seed_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (seed_id, seed_score) in seed_vec {
            if visited.contains(seed_id) || expanded.len() >= limit {
                continue;
            }
            visited.insert(seed_id.clone());

            // Traverse causes (backward)
            if let Ok(causes_result) = graph.get_causes(seed_id, self.config.causal_max_hops) {
                let mut expansions_added = 0;

                for path in causes_result.paths {
                    if expansions_added >= self.config.max_expansions_per_seed {
                        break;
                    }

                    // Skip if confidence too low
                    if path.confidence < self.config.causal_min_confidence {
                        continue;
                    }

                    // Add each memory in path (except seed itself)
                    for (hop, memory_id) in path.memories.iter().enumerate() {
                        if memory_id == seed_id || expanded.contains_key(memory_id) {
                            continue;
                        }

                        // Score decays with hop distance and path confidence
                        let hop_decay = self.config.decay_factor.powi(hop as i32);
                        let expansion_score = seed_score * path.confidence * hop_decay;

                        expanded.insert(memory_id.clone(), expansion_score);
                        expansions_added += 1;

                        if expansions_added >= self.config.max_expansions_per_seed {
                            break;
                        }
                    }
                }
            }

            // Traverse effects (forward)
            if let Ok(effects_result) = graph.get_effects(seed_id, self.config.causal_max_hops) {
                let mut expansions_added = 0;

                for path in effects_result.paths {
                    if expansions_added >= self.config.max_expansions_per_seed {
                        break;
                    }

                    // Skip if confidence too low
                    if path.confidence < self.config.causal_min_confidence {
                        continue;
                    }

                    // Add each memory in path (except seed itself)
                    for (hop, memory_id) in path.memories.iter().enumerate() {
                        if memory_id == seed_id || expanded.contains_key(memory_id) {
                            continue;
                        }

                        // Score decays with hop distance and path confidence
                        let hop_decay = self.config.decay_factor.powi(hop as i32);
                        let expansion_score = seed_score * path.confidence * hop_decay;

                        expanded.insert(memory_id.clone(), expansion_score);
                        expansions_added += 1;

                        if expansions_added >= self.config.max_expansions_per_seed {
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Expand using entity graph traversal
    ///
    /// For each seed result, find entities mentioned and retrieve other
    /// memories mentioning those entities.
    fn expand_entity(
        &self,
        expanded: &mut HashMap<MemoryId, f32>,
        seed_results: &HashMap<MemoryId, f32>,
        limit: usize,
    ) -> Result<()> {
        let graph = self.graph_manager.read().unwrap();

        // Track which seeds we've already expanded
        let mut visited = HashSet::new();

        // Sort seeds by score (highest first)
        let mut seed_vec: Vec<_> = seed_results.iter().collect();
        seed_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (seed_id, seed_score) in seed_vec {
            if visited.contains(seed_id) || expanded.len() >= limit {
                continue;
            }
            visited.insert(seed_id.clone());

            // Get entities linked to this seed memory from the entity graph
            let seed_entities = graph.get_memory_entities(seed_id);

            if !seed_entities.is_empty() {
                let mut expansions_added = 0;

                for entity_id in seed_entities {
                    if expansions_added >= self.config.max_expansions_per_seed {
                        break;
                    }

                    // Query entity graph for other memories mentioning this entity
                    let entity_results = graph.get_entity_memories(&entity_id);

                    for memory_id in entity_results.memories {
                        if &memory_id == seed_id || expanded.contains_key(&memory_id) {
                            continue;
                        }

                        // Score based on seed score with decay
                        // All entity matches get equal weight
                        let expansion_score = seed_score * self.config.decay_factor;

                        expanded.insert(memory_id.clone(), expansion_score);
                        expansions_added += 1;

                        if expansions_added >= self.config.max_expansions_per_seed {
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Expand using hybrid traversal (entity + causal)
    ///
    /// Uses both entity and causal graphs for maximum coverage.
    /// Good for temporal queries where relationships might be implicit.
    fn expand_hybrid(
        &self,
        expanded: &mut HashMap<MemoryId, f32>,
        seed_results: &HashMap<MemoryId, f32>,
        limit: usize,
    ) -> Result<()> {
        // First try entity expansion (cheaper, more coverage)
        self.expand_entity(expanded, seed_results, limit)?;

        // Then try causal expansion if we haven't hit limit
        if expanded.len() < limit {
            self.expand_causal(expanded, seed_results, limit)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traversal_config_default() {
        let config = TraversalConfig::default();
        assert_eq!(config.causal_max_hops, 2);
        assert_eq!(config.entity_max_hops, 1);
        assert_eq!(config.causal_min_confidence, 0.3);
        assert_eq!(config.decay_factor, 0.7);
        assert_eq!(config.max_expansions_per_seed, 5);
    }

    #[test]
    fn test_decay_scoring() {
        let config = TraversalConfig::default();
        let seed_score = 1.0;
        let path_confidence = 0.8;

        // 1-hop decay
        let hop1_score = seed_score * path_confidence * config.decay_factor;
        assert!((hop1_score - 0.56).abs() < 0.01);

        // 2-hop decay
        let hop2_score = seed_score * path_confidence * config.decay_factor.powi(2);
        assert!((hop2_score - 0.392).abs() < 0.01);
    }
}
