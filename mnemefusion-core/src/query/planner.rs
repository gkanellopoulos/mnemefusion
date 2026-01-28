//! Query planner for coordinating multi-dimensional queries
//!
//! The QueryPlanner classifies query intent and coordinates retrieval across
//! semantic, temporal, causal, and entity dimensions with adaptive weighting.

use crate::{
    error::Result,
    graph::GraphManager,
    index::{TemporalIndex, VectorIndex},
    ingest::{get_causal_extractor, get_temporal_extractor, EntityExtractor, SimpleEntityExtractor},
    query::{
        fusion::{FusedResult, FusionEngine},
        intent::{IntentClassification, IntentClassifier},
    },
    storage::StorageEngine,
    types::{Memory, MemoryId, MetadataFilter, Timestamp},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Query planner that coordinates multi-dimensional retrieval
pub struct QueryPlanner {
    pub(crate) storage: Arc<StorageEngine>,
    pub(crate) vector_index: Arc<RwLock<VectorIndex>>,
    pub(crate) bm25_index: Arc<crate::index::BM25Index>,
    pub(crate) temporal_index: Arc<TemporalIndex>,
    pub(crate) graph_manager: Arc<RwLock<GraphManager>>,
    intent_classifier: IntentClassifier,
    fusion_engine: FusionEngine,
    semantic_prefilter_threshold: f32,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(
        storage: Arc<StorageEngine>,
        vector_index: Arc<RwLock<VectorIndex>>,
        bm25_index: Arc<crate::index::BM25Index>,
        temporal_index: Arc<TemporalIndex>,
        graph_manager: Arc<RwLock<GraphManager>>,
        fusion_semantic_threshold: f32,
        semantic_prefilter_threshold: f32,
        fusion_strategy: crate::query::FusionStrategy,
        rrf_k: f32,
    ) -> Self {
        Self {
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            intent_classifier: IntentClassifier::new(),
            fusion_engine: FusionEngine::new()
                .with_semantic_threshold(fusion_semantic_threshold)
                .with_strategy(fusion_strategy)
                .with_rrf_k(rrf_k),
            semantic_prefilter_threshold,
        }
    }

    /// Execute a multi-dimensional query
    ///
    /// Classifies the query intent, retrieves results from relevant dimensions,
    /// and fuses them with adaptive weights.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The natural language query
    /// * `query_embedding` - Vector embedding of the query
    /// * `limit` - Maximum number of results to return
    /// * `namespace` - Optional namespace filter. If provided, only returns memories in this namespace
    /// * `filters` - Optional metadata filters. All filters must match (AND logic)
    ///
    /// # Returns
    ///
    /// Tuple of (intent classification, fused results)
    pub fn query(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
    ) -> Result<(IntentClassification, Vec<FusedResult>)> {
        // Step 1: Classify intent
        let intent = self.intent_classifier.classify(query_text);

        // Step 1.5: Entity-first retrieval path (Sprint 18 Task 18.4)
        // If query has entity_focus, fetch ALL memories mentioning that entity
        // This bypasses top-K semantic search to get complete entity information
        if let Some(entity_name) = intent.entity_focus.clone() {
            return self.retrieve_entity_focused(
                &entity_name,
                query_text,
                query_embedding,
                limit,
                namespace,
                filters,
                intent,
            );
        }

        // Step 2: Retrieve from all dimensions (fetch more to account for filtering)
        let needs_filtering =
            namespace.is_some() || (filters.is_some() && !filters.unwrap().is_empty());
        let fetch_multiplier = if needs_filtering { 5 } else { 3 };
        let mut semantic_scores =
            self.semantic_search(query_embedding, limit * fetch_multiplier)?;
        let mut bm25_scores = self.bm25_search(query_text, limit * fetch_multiplier)?;
        let mut temporal_scores = self.temporal_search(query_text, limit * fetch_multiplier)?;
        let mut causal_scores = self.causal_search(query_text, limit * fetch_multiplier)?;
        let mut entity_scores = self.entity_search(query_text, limit * fetch_multiplier)?;

        // Step 2.3: Pre-fusion semantic filtering (Sprint 18 Task 18.1)
        // Filter out low-quality semantic matches before fusion to improve precision
        if self.semantic_prefilter_threshold > 0.0 {
            semantic_scores.retain(|_id, score| *score >= self.semantic_prefilter_threshold);
        }

        // Step 2.5: Filter by namespace if provided
        if let Some(ns) = namespace {
            self.filter_by_namespace(&mut semantic_scores, ns)?;
            self.filter_by_namespace(&mut bm25_scores, ns)?;
            self.filter_by_namespace(&mut temporal_scores, ns)?;
            self.filter_by_namespace(&mut causal_scores, ns)?;
            self.filter_by_namespace(&mut entity_scores, ns)?;
        }

        // Step 2.6: Filter by metadata if provided
        if let Some(filter_list) = filters {
            if !filter_list.is_empty() {
                self.filter_by_metadata(&mut semantic_scores, filter_list)?;
                self.filter_by_metadata(&mut bm25_scores, filter_list)?;
                self.filter_by_metadata(&mut temporal_scores, filter_list)?;
                self.filter_by_metadata(&mut causal_scores, filter_list)?;
                self.filter_by_metadata(&mut entity_scores, filter_list)?;
            }
        }

        // Step 2.7: Graph traversal expansion for retrieval augmentation (conditional)
        // Only expand if seed results have high quality scores (>0.6 threshold)
        // This prevents noise from over-expansion in session-based retrieval
        let mut seed_results = semantic_scores.clone();
        for (id, score) in &bm25_scores {
            seed_results
                .entry(id.clone())
                .and_modify(|s| *s = (*s).max(*score))
                .or_insert(*score);
        }

        // Calculate average seed quality
        let avg_seed_quality = if !seed_results.is_empty() {
            seed_results.values().sum::<f32>() / seed_results.len() as f32
        } else {
            0.0
        };

        // Only perform graph traversal if seed quality is good (prevents noise)
        const GRAPH_EXPANSION_THRESHOLD: f32 = 0.6;
        if avg_seed_quality >= GRAPH_EXPANSION_THRESHOLD {
            let graph_traversal = crate::query::graph_traversal::GraphTraversal::new(
                self.graph_manager.clone(),
                self.storage.clone(),
                crate::query::graph_traversal::TraversalConfig::default(),
            );

            let expanded_results = graph_traversal.expand(&seed_results, intent.intent, limit * 5)?;

            // Merge expanded results into dimension scores
            // Expanded memories get added to entity/causal scores based on their source
            for (expanded_id, expansion_score) in expanded_results {
                // Skip if already in seed results
                if seed_results.contains_key(&expanded_id) {
                    continue;
                }

                // Add to appropriate dimension score maps based on intent
                match intent.intent {
                    crate::query::QueryIntent::Causal => {
                        causal_scores.insert(expanded_id.clone(), expansion_score);
                    }
                    crate::query::QueryIntent::Entity | crate::query::QueryIntent::Factual => {
                        entity_scores.insert(expanded_id.clone(), expansion_score);
                    }
                    crate::query::QueryIntent::Temporal => {
                        // Hybrid expansion - add to both temporal and entity
                        temporal_scores
                            .entry(expanded_id.clone())
                            .and_modify(|s| *s = (*s).max(expansion_score * 0.7))
                            .or_insert(expansion_score * 0.7);
                        entity_scores.insert(expanded_id, expansion_score * 0.5);
                    }
                }
            }
        }

        // Step 3: Fuse results with adaptive weights
        let mut fused_results = self.fusion_engine.fuse(
            intent.intent,
            &semantic_scores,
            &bm25_scores,
            &temporal_scores,
            &causal_scores,
            &entity_scores,
        );

        // Step 3.5: Cross-dimensional validation (Sprint 18 Task 18.2)
        // Calculate confidence based on how many dimensions contributed to each result
        // Multi-dimensional matches are more reliable than single-dimension matches
        for result in &mut fused_results {
            // Count how many dimensions contributed (score > 0.0)
            let mut dimension_count = 0;

            if result.semantic_score > 0.0 {
                dimension_count += 1;
            }
            if result.bm25_score > 0.0 {
                dimension_count += 1;
            }
            if result.temporal_score > 0.0 {
                dimension_count += 1;
            }
            if result.causal_score > 0.0 {
                dimension_count += 1;
            }
            if result.entity_score > 0.0 {
                dimension_count += 1;
            }

            // Assign confidence based on dimensional coverage
            // More dimensions agreeing = higher confidence = more reliable result
            result.confidence = match dimension_count {
                5 => 1.0,   // All 5 dimensions agree - highest confidence
                4 => 0.9,   // 4 dimensions - very high confidence
                3 => 0.8,   // 3 dimensions - high confidence
                2 => 0.6,   // 2 dimensions - medium confidence
                1 => 0.4,   // 1 dimension only - low confidence (potential noise)
                _ => 0.2,   // 0 dimensions (shouldn't happen) - very low confidence
            };

            // Adjust fused_score by confidence to penalize single-dimension matches
            result.fused_score *= result.confidence;
        }

        // Re-sort by adjusted fused_score (confidence-weighted)
        fused_results.sort_by(|a, b| {
            b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 4: Multi-turn aggregation for list/collection queries
        // Note: Currently has minimal impact but kept for future improvement
        let aggregator = crate::query::aggregator::MultiTurnAggregator::default();
        let query_type = aggregator.classify_query(query_text);
        let final_results = aggregator.aggregate(
            query_type,
            query_text,
            fused_results,
            &self.storage,
            limit,
        )?;

        Ok((intent, final_results))
    }

    /// Entity-first retrieval path (Sprint 18 Task 18.4)
    ///
    /// For entity-focused queries (e.g., "What does Alice like?"), this method:
    /// 1. Looks up the entity by name
    /// 2. Fetches ALL memories mentioning that entity (bypasses top-K limit)
    /// 3. Ranks by semantic similarity + keyword matching
    /// 4. Applies filters and returns top results
    ///
    /// This solves Category 1 failures where evidence is scattered across many turns.
    fn retrieve_entity_focused(
        &self,
        entity_name: &str,
        query_text: &str,
        query_embedding: &[f32],
        limit: usize,
        namespace: Option<&str>,
        filters: Option<&[MetadataFilter]>,
        intent: IntentClassification,
    ) -> Result<(IntentClassification, Vec<FusedResult>)> {
        // Step 1: Find entity by name (case-insensitive)
        let entity = self.storage.find_entity_by_name(entity_name)?;

        if entity.is_none() {
            // Entity not found - return empty results
            return Ok((intent, vec![]));
        }

        let entity = entity.unwrap();

        // Step 2: Get ALL memories mentioning this entity from the graph
        let graph = self.graph_manager.read().unwrap();
        let entity_result = graph.get_entity_memories(&entity.id);
        let memory_ids = entity_result.memories;
        drop(graph); // Release lock

        if memory_ids.is_empty() {
            return Ok((intent, vec![]));
        }

        // Step 3: Calculate scores for each memory
        let mut scored_results = Vec::new();

        for memory_id in memory_ids {
            // Get memory from storage
            let memory = match self.storage.get_memory_by_u64(memory_id.to_u64())? {
                Some(m) => m,
                None => continue, // Skip if memory not found
            };

            // Apply namespace filter
            if let Some(ns) = namespace {
                if memory.get_namespace() != ns {
                    continue;
                }
            }

            // Apply metadata filters
            if let Some(filter_list) = filters {
                if !filter_list.is_empty() {
                    if !Self::memory_matches_filters(&memory, filter_list) {
                        continue;
                    }
                }
            }

            // Calculate semantic similarity
            let semantic_score = if !memory.embedding.is_empty() && query_embedding.len() == memory.embedding.len() {
                cosine_similarity(query_embedding, &memory.embedding)
            } else {
                0.0
            };

            // Calculate BM25 score for keyword matching
            let bm25_score = {
                let results = self.bm25_index.search(query_text, 1000)?;
                results.iter()
                    .find(|r| r.memory_id == memory_id)
                    .map(|r| r.score)
                    .unwrap_or(0.0)
            };

            // Combined score: prioritize semantic similarity with keyword boost
            // For entity queries, we want memories that mention the entity AND match query semantics
            let combined_score = 0.7 * semantic_score + 0.3 * bm25_score;

            scored_results.push(FusedResult {
                id: memory_id.clone(),
                semantic_score,
                bm25_score,
                temporal_score: 0.0,
                causal_score: 0.0,
                entity_score: 1.0, // All results mention the entity
                fused_score: combined_score,
                confidence: 0.9, // High confidence since we know the entity is mentioned
            });
        }

        // Step 4: Sort by combined score and take top-K
        scored_results.sort_by(|a, b| {
            b.fused_score.partial_cmp(&a.fused_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        let final_results = scored_results.into_iter().take(limit).collect();

        Ok((intent, final_results))
    }

    /// Perform semantic similarity search
    fn semantic_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<HashMap<MemoryId, f32>> {
        let index = self.vector_index.read().unwrap();
        let results = index.search(query_embedding, limit)?;

        let mut scores = HashMap::new();
        for result in results {
            scores.insert(result.id, result.similarity);
        }

        Ok(scores)
    }

    /// Perform BM25 keyword search
    ///
    /// Searches using BM25 algorithm for exact term matching.
    /// Returns empty if query has no valid terms (too short or all stopwords).
    fn bm25_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let results = self.bm25_index.search(query_text, limit)?;

        let mut scores = HashMap::new();
        for result in results {
            scores.insert(result.memory_id, result.score);
        }

        // Normalize scores to 0.0-1.0 range
        FusionEngine::normalize_scores(&mut scores);

        Ok(scores)
    }

    /// Perform temporal search based on temporal content matching
    ///
    /// Extracts temporal expressions from query and matches them to temporal expressions
    /// in memory content. Falls back to weak recency signal if query has no temporal context.
    fn temporal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        // Sprint 18 Task 18.6: Use temporal content scoring
        // Match temporal expressions in query to temporal expressions in memory content
        let results = self.temporal_index.search_temporal_content(query_text, limit)?;

        // If we found temporal matches, return them
        if !results.is_empty() {
            let mut scores: HashMap<MemoryId, f32> = results.into_iter().collect();

            // Normalize scores to 0.0-1.0 range
            FusionEngine::normalize_scores(&mut scores);

            Ok(scores)
        } else {
            // No temporal matches found - fall back to weak recency signal
            self.temporal_search_recency_fallback(limit)
        }
    }

    /// Fallback temporal search using weak recency signal
    ///
    /// Used when query has no temporal context or no temporal matches found.
    /// Provides a weak signal (scores scaled down to 0.0-0.3 range) to avoid
    /// biasing non-temporal queries toward recent memories.
    fn temporal_search_recency_fallback(&self, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let results = self.temporal_index.recent(limit)?;

        let mut scores = HashMap::new();
        let count = results.len();

        for (i, result) in results.into_iter().enumerate() {
            // Linear decay from 0.3 to 0.0 (weak signal)
            let score = if count > 1 {
                0.3 * (1.0 - (i as f32 / (count - 1) as f32))
            } else {
                0.3
            };
            scores.insert(result.id, score);
        }

        Ok(scores)
    }

    /// Perform entity-based search using content matching
    ///
    /// Extracts entities from query using SimpleEntityExtractor (with stop word filtering)
    /// and matches them to entities in memory content. Returns empty if query has no entities.
    fn entity_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let entity_extractor = SimpleEntityExtractor::new();

        // Extract entities from query (with stop word filtering)
        let query_entities = entity_extractor.extract(query_text)?;

        // If query has no entities, return empty (no entity scoring)
        if query_entities.is_empty() {
            return Ok(HashMap::new());
        }

        // Convert query entities to lowercase set for case-insensitive matching
        let query_entity_set: std::collections::HashSet<String> =
            query_entities.iter().map(|e| e.to_lowercase()).collect();

        // Get recent memories to check their entity metadata
        // Fetch more to find matches
        let recent_memories = self.temporal_index.recent(limit * 10)?;

        let mut scores = HashMap::new();

        for temporal_result in recent_memories {
            // Load memory to access metadata
            if let Some(memory) = self.storage.get_memory(&temporal_result.id)? {
                // Get entity names from memory metadata
                if let Some(entities_json) = memory.get_metadata("entity_names") {
                    // Parse JSON array of entity names
                    if let Ok(memory_entities) = serde_json::from_str::<Vec<String>>(entities_json)
                    {
                        // Calculate overlap score
                        let overlap_score =
                            Self::calculate_entity_overlap(&query_entity_set, &memory_entities);

                        if overlap_score > 0.0 {
                            scores.insert(temporal_result.id, overlap_score);
                        }
                    }
                }
            }

            // Stop once we have enough scored results
            if scores.len() >= limit * 2 {
                break;
            }
        }

        // If we found entity matches, return them
        if !scores.is_empty() {
            // Normalize scores to 0.0-1.0 range
            FusionEngine::normalize_scores(&mut scores);

            // Take top results by score
            let mut score_vec: Vec<_> = scores.into_iter().collect();
            score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            score_vec.truncate(limit);

            Ok(score_vec.into_iter().collect())
        } else {
            // No entity matches found
            Ok(HashMap::new())
        }
    }

    /// Calculate entity overlap score between query and memory entities
    ///
    /// Returns score from 0.0 to 1.0 based on:
    /// - Exact matches (case-insensitive): 1.0 per match
    /// - Partial matches (substring): 0.5 per match
    /// - Average across query entities
    fn calculate_entity_overlap(
        query_entity_set: &std::collections::HashSet<String>,
        memory_entities: &[String],
    ) -> f32 {
        if query_entity_set.is_empty() || memory_entities.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0;

        // Convert memory entities to lowercase for comparison
        let memory_entity_set: std::collections::HashSet<String> =
            memory_entities.iter().map(|e| e.to_lowercase()).collect();

        for query_entity in query_entity_set {
            // Check for exact match (case-insensitive)
            if memory_entity_set.contains(query_entity) {
                total_score += 1.0;
            } else {
                // Check for partial match (substring)
                for memory_entity in &memory_entity_set {
                    if memory_entity.contains(query_entity) || query_entity.contains(memory_entity)
                    {
                        total_score += 0.5;
                        break;
                    }
                }
            }
        }

        // Average score across query entities
        if query_entity_set.len() > 0 {
            total_score / query_entity_set.len() as f32
        } else {
            0.0
        }
    }

    /// Perform causal language-based search
    ///
    /// Checks if query has causal intent. If yes, scores memories based on
    /// causal language density. If no, returns empty (no causal scoring).
    fn causal_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let causal_extractor = get_causal_extractor();

        // Check if query has causal intent
        if !causal_extractor.has_causal_intent(query_text) {
            // No causal focus → no causal scoring
            return Ok(HashMap::new());
        }

        // Get recent memories to check their causal metadata
        // Fetch more to find matches
        let recent_memories = self.temporal_index.recent(limit * 10)?;

        let mut scores = HashMap::new();

        for temporal_result in recent_memories {
            // Load memory to access metadata
            if let Some(memory) = self.storage.get_memory(&temporal_result.id)? {
                // Get causal density from memory metadata
                if let Some(density_str) = memory.get_metadata("causal_density") {
                    if let Ok(causal_density) = density_str.parse::<f32>() {
                        // Only score memories with significant causal density (> 0.1)
                        if causal_density > 0.1 {
                            // Check if memory has causal graph links for boost
                            let has_graph_links = {
                                let graph = self.graph_manager.read().unwrap();
                                graph.get_causes(&temporal_result.id, 1).ok().map_or(false, |r| !r.paths.is_empty())
                                    || graph.get_effects(&temporal_result.id, 1).ok().map_or(false, |r| !r.paths.is_empty())
                            };

                            // Calculate relevance score with optional graph boost
                            let score =
                                causal_extractor.calculate_relevance_score(causal_density, has_graph_links);

                            if score > 0.0 {
                                scores.insert(temporal_result.id, score);
                            }
                        }
                    }
                }
            }

            // Stop once we have enough scored results
            if scores.len() >= limit * 2 {
                break;
            }
        }

        // If we found causal matches, return them
        if !scores.is_empty() {
            // Normalize scores to 0.0-1.0 range
            FusionEngine::normalize_scores(&mut scores);

            // Take top results by score
            let mut score_vec: Vec<_> = scores.into_iter().collect();
            score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            score_vec.truncate(limit);

            Ok(score_vec.into_iter().collect())
        } else {
            // No causal matches found
            Ok(HashMap::new())
        }
    }

    /// Perform temporal range search
    ///
    /// Search memories within a specific time range
    ///
    /// # Arguments
    ///
    /// * `start` - Start of time range
    /// * `end` - End of time range
    /// * `limit` - Maximum number of results
    /// * `namespace` - Optional namespace filter
    pub fn temporal_range_query(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<FusedResult>> {
        // Fetch more results if filtering by namespace
        let fetch_limit = if namespace.is_some() {
            limit * 3
        } else {
            limit
        };
        let results = self.temporal_index.range_query(start, end, fetch_limit)?;

        // Convert to fused results with temporal scores
        let mut fused: Vec<FusedResult> = Vec::new();
        let mut position = 0;

        for result in results {
            // Filter by namespace if provided
            if let Some(ns) = namespace {
                if let Ok(Some(memory)) = self.storage.get_memory(&result.id) {
                    if memory.get_namespace() != ns {
                        continue; // Skip memories not in this namespace
                    }
                } else {
                    continue; // Skip if memory not found
                }
            }

            // Calculate temporal score based on position
            let temporal_score = if fetch_limit > 1 {
                1.0 - (position as f32 / (fetch_limit - 1) as f32)
            } else {
                1.0
            };

            fused.push(FusedResult {
                id: result.id,
                semantic_score: 0.0,
                bm25_score: 0.0,
                temporal_score,
                causal_score: 0.0,
                entity_score: 0.0,
                fused_score: temporal_score,
                confidence: 1.0,
            });

            position += 1;
            if fused.len() >= limit {
                break;
            }
        }

        Ok(fused)
    }

    /// Filter a score map by namespace
    ///
    /// Removes entries whose memories are not in the specified namespace
    fn filter_by_namespace(
        &self,
        scores: &mut HashMap<MemoryId, f32>,
        namespace: &str,
    ) -> Result<()> {
        // Collect IDs to remove (can't remove during iteration)
        let mut to_remove = Vec::new();

        for memory_id in scores.keys() {
            if let Some(memory) = self.storage.get_memory(memory_id)? {
                if memory.get_namespace() != namespace {
                    to_remove.push(memory_id.clone());
                }
            } else {
                // Memory doesn't exist, remove it
                to_remove.push(memory_id.clone());
            }
        }

        // Remove filtered-out IDs
        for id in to_remove {
            scores.remove(&id);
        }

        Ok(())
    }

    /// Filter a score map by metadata filters
    ///
    /// Removes entries whose memories don't match ALL provided filters
    fn filter_by_metadata(
        &self,
        scores: &mut HashMap<MemoryId, f32>,
        filters: &[MetadataFilter],
    ) -> Result<()> {
        if filters.is_empty() {
            return Ok(());
        }

        // Collect IDs to remove (can't remove during iteration)
        let mut to_remove = Vec::new();

        for memory_id in scores.keys() {
            if let Some(memory) = self.storage.get_memory(memory_id)? {
                // Check if memory matches ALL filters
                if !Self::memory_matches_filters(&memory, filters) {
                    to_remove.push(memory_id.clone());
                }
            } else {
                // Memory doesn't exist, remove it
                to_remove.push(memory_id.clone());
            }
        }

        // Remove filtered-out IDs
        for id in to_remove {
            scores.remove(&id);
        }

        Ok(())
    }

    /// Check if a memory matches all metadata filters
    fn memory_matches_filters(memory: &Memory, filters: &[MetadataFilter]) -> bool {
        for filter in filters {
            let value = memory.metadata.get(&filter.field).map(|s| s.as_str());
            if !filter.matches(value) {
                return false; // Any filter fails = memory doesn't match
            }
        }
        true // All filters passed
    }
}

/// Calculate cosine similarity between two vectors
///
/// Returns similarity in range [0.0, 1.0] where 1.0 is identical vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    (dot_product / (magnitude_a * magnitude_b)).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::GraphManager,
        index::{VectorIndex, VectorIndexConfig},
        storage::StorageEngine,
        types::Memory,
    };
    use tempfile::tempdir;

    fn create_test_planner() -> (QueryPlanner, tempfile::TempDir) {
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

        let bm25_index = Arc::new(crate::index::BM25Index::new(
            Arc::clone(&storage),
            crate::index::BM25Config::default(),
        ));

        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));
        let graph_manager = Arc::new(RwLock::new(GraphManager::new()));

        // Use 0.0 thresholds for tests to avoid filtering test results
        // Use Weighted strategy for tests to maintain existing test expectations
        let planner = QueryPlanner::new(
            storage,
            vector_index,
            bm25_index,
            temporal_index,
            graph_manager,
            0.0, // fusion_semantic_threshold
            0.0, // semantic_prefilter_threshold
            crate::query::FusionStrategy::Weighted, // strategy (weighted for backward compat in tests)
            60.0, // rrf_k (not used with Weighted strategy)
        );

        (planner, dir)
    }

    #[test]
    fn test_query_planner_creation() {
        let (_planner, _dir) = create_test_planner();
        // Just verify it creates without panic
    }

    #[test]
    fn test_semantic_search() {
        let (planner, _dir) = create_test_planner();

        // Add a memory
        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let mem_id = memory.id.clone();
        planner.storage.store_memory(&memory).unwrap();
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem_id.clone(), &memory.embedding).unwrap();
        }

        // Search
        let scores = planner.semantic_search(&vec![0.1; 384], 10).unwrap();
        assert!(!scores.is_empty());
        // The scores map should contain at least one result
        assert!(scores.len() > 0);
    }

    #[test]
    fn test_temporal_search() {
        let (planner, _dir) = create_test_planner();

        // Add memories
        let mem1 = Memory::new("Memory 1".to_string(), vec![0.1; 384]);
        let mem2 = Memory::new("Memory 2".to_string(), vec![0.2; 384]);

        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Also add to temporal index (normally done by ingestion pipeline)
        planner
            .temporal_index
            .add(&mem1.id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2.id, mem2.created_at)
            .unwrap();

        // Search with query text (no temporal expressions = fallback to recency)
        let scores = planner.temporal_search("test query", 10).unwrap();
        assert_eq!(scores.len(), 2);

        // With fallback, most recent should have higher score (but weak signal)
        let score1 = scores[&mem1.id];
        let score2 = scores[&mem2.id];
        assert!(score2 >= score1); // mem2 was added later
    }

    #[test]
    fn test_entity_search() {
        let (planner, _dir) = create_test_planner();

        // This test just verifies entity_search doesn't crash with no entities
        let scores = planner.entity_search("Show me Alice", 10).unwrap();
        // No entities exist, so should be empty
        assert_eq!(scores.len(), 0);
    }

    #[test]
    fn test_full_query() {
        let (planner, _dir) = create_test_planner();

        // Add a memory
        let memory = Memory::new("Test content".to_string(), vec![0.1; 384]);
        let mem_id = memory.id.clone();
        planner.storage.store_memory(&memory).unwrap();
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem_id.clone(), &memory.embedding).unwrap();
        }
        // Add to temporal index too
        planner
            .temporal_index
            .add(&mem_id, memory.created_at)
            .unwrap();

        // Execute query
        let (intent, results) = planner
            .query("test query", &vec![0.1; 384], 10, None, None)
            .unwrap();

        // Should classify as factual
        assert_eq!(intent.intent, crate::query::intent::QueryIntent::Factual);

        // Should have results
        assert!(!results.is_empty());
        // Results should contain our memory (may not be first if other memories exist)
        assert!(results.iter().any(|r| r.id == mem_id));
    }

    #[test]
    fn test_temporal_range_query() {
        let (planner, _dir) = create_test_planner();

        let now = Timestamp::now();
        let mem1 = Memory::new_with_timestamp(
            "Memory 1".to_string(),
            vec![0.1; 384],
            now.subtract_days(1),
        );

        planner.storage.store_memory(&mem1).unwrap();
        // Add to temporal index
        planner
            .temporal_index
            .add(&mem1.id, mem1.created_at)
            .unwrap();

        let results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, mem1.id);
    }

    #[test]
    fn test_query_with_namespace_filtering() {
        let (planner, _dir) = create_test_planner();

        // Create memories in different namespaces
        let mut mem1 = Memory::new("NS1 memory".to_string(), vec![0.1; 384]);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("NS2 memory".to_string(), vec![0.15; 384]);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        let mem3 = Memory::new("Default memory".to_string(), vec![0.2; 384]);
        let mem3_id = mem3.id.clone();

        // Store all memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Add to indexes
        {
            let mut index = planner.vector_index.write().unwrap();
            index.add(mem1_id.clone(), &mem1.embedding).unwrap();
            index.add(mem2_id.clone(), &mem2.embedding).unwrap();
            index.add(mem3_id.clone(), &mem3.embedding).unwrap();
        }
        planner
            .temporal_index
            .add(&mem1_id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2_id, mem2.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem3_id, mem3.created_at)
            .unwrap();

        // Query with ns1 filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns1"), None)
            .unwrap();

        // Should only contain mem1
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id == mem1_id));

        // Query with ns2 filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns2"), None)
            .unwrap();

        // Should only contain mem2
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id == mem2_id));

        // Query with default namespace filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some(""), None)
            .unwrap();

        // Should only contain mem3
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id == mem3_id));
    }

    #[test]
    fn test_temporal_range_query_with_namespace() {
        let (planner, _dir) = create_test_planner();

        let now = Timestamp::now();
        let ts1 = now.subtract_days(1);
        // Use slightly different timestamp for mem2 to avoid any potential uniqueness issues
        let ts2 = Timestamp::from_unix_secs(ts1.as_unix_secs() + 60.0); // 1 minute later

        // Create memories in different namespaces
        let mut mem1 = Memory::new_with_timestamp("NS1 memory".to_string(), vec![0.1; 384], ts1);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new_with_timestamp("NS2 memory".to_string(), vec![0.2; 384], ts2);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        // Store and index
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner
            .temporal_index
            .add(&mem1_id, mem1.created_at)
            .unwrap();
        planner
            .temporal_index
            .add(&mem2_id, mem2.created_at)
            .unwrap();

        // First verify memories are stored correctly
        let retrieved_mem1 = planner.storage.get_memory(&mem1_id).unwrap().unwrap();
        assert_eq!(retrieved_mem1.get_namespace(), "ns1");

        // Query without namespace filter first
        let all_results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, None)
            .unwrap();
        assert_eq!(
            all_results.len(),
            2,
            "Should find both memories without filter"
        );

        // Query with namespace filter
        let results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, Some("ns1"))
            .unwrap();

        // Should only return mem1
        assert_eq!(results.len(), 1, "Should find exactly one memory in ns1");
        assert_eq!(results[0].id, mem1_id);
    }

    #[test]
    fn test_filter_by_namespace() {
        let (planner, _dir) = create_test_planner();

        // Create memories in different namespaces
        let mut mem1 = Memory::new("NS1 memory".to_string(), vec![0.1; 384]);
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("NS2 memory".to_string(), vec![0.2; 384]);
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);

        // Filter by ns1
        planner.filter_by_namespace(&mut scores, "ns1").unwrap();

        // Should only contain mem1
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
        assert!(!scores.contains_key(&mem2_id));
    }

    #[test]
    fn test_filter_by_metadata_exact_match() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different metadata
        let mut mem1 = Memory::new("Event memory".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Task memory".to_string(), vec![0.2; 384]);
        mem2.metadata.insert("type".to_string(), "task".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        let mem2_id = mem2.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);

        // Filter by type=event
        let filters = vec![MetadataFilter::eq("type", "event")];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should only contain mem1
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
        assert!(!scores.contains_key(&mem2_id));
    }

    #[test]
    fn test_filter_by_metadata_multiple_filters() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different metadata
        let mut mem1 = Memory::new("High priority event".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        mem1.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Low priority event".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        mem2.metadata
            .insert("priority".to_string(), "low".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("High priority task".to_string(), vec![0.3; 384]);
        mem3.metadata.insert("type".to_string(), "task".to_string());
        mem3.metadata
            .insert("priority".to_string(), "high".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by type=event AND priority=high
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should only contain mem1 (only event with high priority)
        assert_eq!(scores.len(), 1);
        assert!(scores.contains_key(&mem1_id));
    }

    #[test]
    fn test_filter_by_metadata_comparison_operators() {
        let (planner, _dir) = create_test_planner();

        // Create memories with numeric priority values
        let mut mem1 = Memory::new("Priority 8".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("priority".to_string(), "8".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Priority 5".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("priority".to_string(), "5".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("Priority 3".to_string(), vec![0.3; 384]);
        mem3.metadata
            .insert("priority".to_string(), "3".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by priority >= "5"
        let filters = vec![MetadataFilter::gte("priority", "5")];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should contain mem1 and mem2 (priority 8 and 5)
        assert_eq!(scores.len(), 2);
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));
        assert!(!scores.contains_key(&mem3_id));
    }

    #[test]
    fn test_filter_by_metadata_in_operator() {
        let (planner, _dir) = create_test_planner();

        // Create memories with different categories
        let mut mem1 = Memory::new("Food memory".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("category".to_string(), "food".to_string());
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new("Travel memory".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("category".to_string(), "travel".to_string());
        let mem2_id = mem2.id.clone();

        let mut mem3 = Memory::new("Work memory".to_string(), vec![0.3; 384]);
        mem3.metadata
            .insert("category".to_string(), "work".to_string());
        let mem3_id = mem3.id.clone();

        // Store memories
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.storage.store_memory(&mem3).unwrap();

        // Create score map
        let mut scores = HashMap::new();
        scores.insert(mem1_id.clone(), 0.9);
        scores.insert(mem2_id.clone(), 0.8);
        scores.insert(mem3_id.clone(), 0.7);

        // Filter by category IN ["food", "travel"]
        let filters = vec![MetadataFilter::in_list(
            "category",
            vec!["food".to_string(), "travel".to_string()],
        )];
        planner.filter_by_metadata(&mut scores, &filters).unwrap();

        // Should contain mem1 and mem2
        assert_eq!(scores.len(), 2);
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));
        assert!(!scores.contains_key(&mem3_id));
    }

    #[test]
    fn test_memory_matches_filters() {
        // Test exact match
        let mut memory = Memory::new("Test".to_string(), vec![0.1; 384]);
        memory
            .metadata
            .insert("type".to_string(), "event".to_string());

        let filters = vec![MetadataFilter::eq("type", "event")];
        assert!(QueryPlanner::memory_matches_filters(&memory, &filters));

        let filters = vec![MetadataFilter::eq("type", "task")];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));

        // Test missing field
        let filters = vec![MetadataFilter::eq("priority", "high")];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));

        // Test multiple filters (AND logic)
        memory
            .metadata
            .insert("priority".to_string(), "high".to_string());
        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "high"),
        ];
        assert!(QueryPlanner::memory_matches_filters(&memory, &filters));

        let filters = vec![
            MetadataFilter::eq("type", "event"),
            MetadataFilter::eq("priority", "low"),
        ];
        assert!(!QueryPlanner::memory_matches_filters(&memory, &filters));
    }

    #[test]
    fn test_temporal_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        // Create ingestion pipeline to extract temporal expressions
        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false, // Disable entity extraction for this test
        );

        // Add memories with temporal expressions
        let mem1 = Memory::new(
            "We had a meeting yesterday about the project".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "The conference was on June 15th, 2023".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Machine learning is a fascinating field".to_string(), // No temporal expression
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with temporal expression matching mem1
        let scores = planner
            .temporal_search("What happened yesterday?", 10)
            .unwrap();

        // Should find mem1 (has "yesterday")
        assert!(scores.contains_key(&mem1_id), "Should find memory with 'yesterday'");

        // mem1 should have higher score than mem3 (no temporal expression)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score3 = scores.get(&mem3_id).unwrap_or(&0.0);
        assert!(
            score1 > score3,
            "Memory with matching temporal expression should score higher than non-temporal memory"
        );
    }

    #[test]
    fn test_temporal_fallback_to_recency() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memories without temporal expressions
        let mem1 = Memory::new(
            "Machine learning techniques".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let mem2 = Memory::new(
            "Deep learning models".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Query without temporal expression
        let scores = planner
            .temporal_search("Tell me about AI", 10)
            .unwrap();

        // Should fall back to weak recency signal
        // Both memories should have scores (recency fallback)
        assert!(scores.contains_key(&mem1_id));
        assert!(scores.contains_key(&mem2_id));

        // Scores should be weak (0.0-0.3 range)
        let score1 = scores.get(&mem1_id).unwrap();
        let score2 = scores.get(&mem2_id).unwrap();
        assert!(
            *score1 <= 0.31 && *score2 <= 0.31,
            "Fallback scores should be weak (≤ 0.3)"
        );

        // More recent memory should have slightly higher score
        assert!(
            score2 >= score1,
            "More recent memory should have higher fallback score"
        );
    }

    #[test]
    fn test_temporal_content_matching_absolute_dates() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with absolute date
        let mem1 = Memory::new(
            "The conference was scheduled for June 15th, 2023".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "We met in May 2024 to discuss plans".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Query with date reference
        let scores = planner
            .temporal_search("When was the June conference?", 10)
            .unwrap();

        // Should find mem1 (has "June")
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with June date"
        );

        // mem1 should score higher than mem2 (different month)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score2 = scores.get(&mem2_id).unwrap_or(&0.0);
        assert!(
            score1 >= score2,
            "Memory with matching month should score higher or equal to non-matching month"
        );
    }

    #[test]
    fn test_entity_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        // Create ingestion pipeline with entity extraction enabled
        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true, // Enable entity extraction
        );

        // Add memories with entities
        let mem1 = Memory::new(
            "Alice presented Project Alpha at the conference".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "Bob worked on Project Beta for Acme Corp".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Machine learning is fascinating".to_string(), // No proper entities
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with entity matching mem1
        let scores = planner
            .entity_search("Tell me about Alice and Project Alpha", 10)
            .unwrap();

        // Should find mem1 (has Alice and Project Alpha)
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with matching entities"
        );

        // mem1 should have higher score than mem3 (no entities)
        let score1 = scores.get(&mem1_id).unwrap_or(&0.0);
        let score3 = scores.get(&mem3_id).unwrap_or(&0.0);
        assert!(
            score1 > score3,
            "Memory with matching entities should score higher than non-entity memory"
        );
    }

    #[test]
    fn test_entity_search_filters_stop_words() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true,
        );

        // Add memory with entities
        let mem1 = Memory::new(
            "Alice met Bob at Building C".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with stop words - should filter them out
        // "The" and "What" are stop words and shouldn't match
        let scores = planner
            .entity_search("What about Alice and The meeting?", 10)
            .unwrap();

        // Should find mem1 (Alice is a real entity)
        // Stop words "What" and "The" should be filtered out during extraction
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with Alice, ignoring stop words"
        );
    }

    #[test]
    fn test_entity_search_empty_when_no_entities() {
        let (planner, _dir) = create_test_planner();

        // Query without entities (all lowercase, no proper nouns)
        let scores = planner
            .entity_search("tell me about machine learning", 10)
            .unwrap();

        // Should return empty (no entities in query)
        assert_eq!(
            scores.len(),
            0,
            "Should return empty when query has no entities"
        );
    }

    #[test]
    fn test_entity_search_partial_match() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true,
        );

        // Add memory with multi-word entity
        let mem1 = Memory::new(
            "Project Alpha is managed by the team".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with partial entity match
        let scores = planner.entity_search("What is Project doing?", 10).unwrap();

        // Should find mem1 (partial match: "Project" in "Project Alpha")
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with partial entity match"
        );
    }

    #[test]
    fn test_entity_overlap_calculation() {
        // Test exact match
        let query_set: std::collections::HashSet<String> =
            vec!["alice".to_string(), "bob".to_string()]
                .into_iter()
                .collect();

        let memory_entities = vec!["Alice".to_string(), "Bob".to_string()];

        let score =
            QueryPlanner::calculate_entity_overlap(&query_set, &memory_entities);

        // Both entities match exactly (case-insensitive)
        assert!((score - 1.0).abs() < 0.01, "Exact match should score 1.0");

        // Test partial match
        let query_set2: std::collections::HashSet<String> =
            vec!["alice".to_string()].into_iter().collect();

        let memory_entities2 = vec!["Bob".to_string()];

        let score2 =
            QueryPlanner::calculate_entity_overlap(&query_set2, &memory_entities2);

        // No match
        assert_eq!(score2, 0.0, "No match should score 0.0");

        // Test substring match
        let query_set3: std::collections::HashSet<String> =
            vec!["project".to_string()].into_iter().collect();

        let memory_entities3 = vec!["Project Alpha".to_string()];

        let score3 =
            QueryPlanner::calculate_entity_overlap(&query_set3, &memory_entities3);

        // Substring match should score 0.5
        assert!(
            (score3 - 0.5).abs() < 0.01,
            "Substring match should score 0.5"
        );
    }

    #[test]
    fn test_causal_content_matching() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with causal language
        let mem1 = Memory::new(
            "The meeting was cancelled because Alice was sick".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Add memory with high causal density
        let mem2 = Memory::new(
            "The bug was caused by a race condition which led to crashes".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        // Add memory without causal language
        let mem3 = Memory::new(
            "We had a nice lunch today".to_string(),
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Query with causal intent
        let scores = planner
            .causal_search("Why was the meeting cancelled?", 10)
            .unwrap();

        // Should find mem1 and mem2 (have causal language)
        assert!(
            scores.contains_key(&mem1_id) || scores.contains_key(&mem2_id),
            "Should find memories with causal language"
        );

        // mem3 should not be in results (no causal language)
        assert!(
            !scores.contains_key(&mem3_id) || scores.get(&mem3_id).unwrap_or(&1.0) < &0.01,
            "Should not score non-causal memory highly"
        );
    }

    #[test]
    fn test_causal_search_empty_when_no_intent() {
        let (planner, _dir) = create_test_planner();

        // Query without causal intent
        let scores = planner
            .causal_search("Tell me about machine learning", 10)
            .unwrap();

        // Should return empty (no causal intent)
        assert_eq!(
            scores.len(),
            0,
            "Should return empty when query has no causal intent"
        );
    }

    #[test]
    fn test_causal_search_with_causal_intent() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with causal explanation
        let mem1 = Memory::new(
            "The server crashed because of a memory leak. This was caused by unclosed connections.".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with "why" - clear causal intent
        let scores = planner
            .causal_search("Why did the server crash?", 10)
            .unwrap();

        // Should find mem1 (has causal explanation)
        assert!(
            scores.contains_key(&mem1_id),
            "Should find memory with causal explanation for 'why' question"
        );

        // Score should be significant
        let score = scores.get(&mem1_id).unwrap();
        assert!(
            *score > 0.0,
            "Causal memory should have positive score"
        );
    }

    #[test]
    fn test_causal_search_density_threshold() {
        use crate::ingest::IngestionPipeline;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            false,
        );

        // Add memory with very low causal density (one marker in long text)
        let mem1 = Memory::new(
            "This is a very long memory with lots of words that do not have causal markers except for this one because word.".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        // Query with causal intent
        let scores = planner
            .causal_search("Why did this happen?", 10)
            .unwrap();

        // Should not find mem1 (causal density too low, < 0.1 threshold)
        assert!(
            !scores.contains_key(&mem1_id),
            "Should filter out memories with very low causal density"
        );
    }

    // Sprint 18 Task 18.4: Entity-first retrieval path tests
    #[test]
    fn test_entity_focused_retrieval() {
        use crate::ingest::IngestionPipeline;
        use crate::Entity;

        let (planner, _dir) = create_test_planner();

        let pipeline = IngestionPipeline::new(
            Arc::clone(&planner.storage),
            Arc::clone(&planner.vector_index),
            Arc::clone(&planner.bm25_index),
            Arc::clone(&planner.temporal_index),
            Arc::clone(&planner.graph_manager),
            true, // Enable entity extraction
        );

        // Create and store entity
        let entity = Entity::new("Alice");
        planner.storage.store_entity(&entity).unwrap();

        // Add memories mentioning Alice (scattered across turns)
        let mem1 = Memory::new(
            "Alice likes playing tennis on weekends".to_string(),
            vec![0.1; 384],
        );
        let mem1_id = mem1.id.clone();
        pipeline.add(mem1).unwrap();

        let mem2 = Memory::new(
            "Alice enjoys reading science fiction books".to_string(),
            vec![0.2; 384],
        );
        let mem2_id = mem2.id.clone();
        pipeline.add(mem2).unwrap();

        let mem3 = Memory::new(
            "Alice prefers coffee over tea".to_string(),
            vec![0.3; 384],
        );
        let mem3_id = mem3.id.clone();
        pipeline.add(mem3).unwrap();

        // Unrelated memory
        let mem4 = Memory::new(
            "Bob works on machine learning projects".to_string(),
            vec![0.4; 384],
        );
        pipeline.add(mem4).unwrap();

        // Link memories to entity manually (since entity extraction is being tested separately)
        {
            let mut graph = planner.graph_manager.write().unwrap();
            graph.link_memory_to_entity(&mem1_id, &entity.id);
            graph.link_memory_to_entity(&mem2_id, &entity.id);
            graph.link_memory_to_entity(&mem3_id, &entity.id);
        }

        // Query with entity-focused pattern (should trigger entity-first retrieval)
        let query_embedding = vec![0.15; 384];
        let (intent, results) = planner
            .query("What does Alice like?", &query_embedding, 10, None, None)
            .unwrap();

        // Should classify as Entity intent
        assert_eq!(intent.intent, crate::query::QueryIntent::Entity);

        // Should extract "Alice" as entity_focus
        assert_eq!(intent.entity_focus, Some("Alice".to_string()));

        // Should retrieve ALL memories mentioning Alice (not limited by top-K semantic)
        assert!(results.len() >= 3, "Should retrieve all memories mentioning Alice");

        // Verify all Alice memories are in results
        let result_ids: Vec<_> = results.iter().map(|r| &r.id).collect();
        assert!(result_ids.contains(&&mem1_id), "Should include mem1 (tennis)");
        assert!(result_ids.contains(&&mem2_id), "Should include mem2 (books)");
        assert!(result_ids.contains(&&mem3_id), "Should include mem3 (coffee)");

        // All results should have high entity_score (1.0 since they mention the entity)
        for result in &results {
            if result_ids.contains(&&result.id) {
                assert_eq!(
                    result.entity_score, 1.0,
                    "All entity-focused results should have entity_score=1.0"
                );
            }
        }
    }

    #[test]
    fn test_entity_focused_retrieval_not_found() {
        let (planner, _dir) = create_test_planner();

        // Query for non-existent entity
        let query_embedding = vec![0.1; 384];
        let (_intent, results) = planner
            .query("What does NonExistentEntity like?", &query_embedding, 10, None, None)
            .unwrap();

        // Should return empty results
        assert_eq!(results.len(), 0, "Should return empty for non-existent entity");
    }
}
