//! Query planner for coordinating multi-dimensional queries
//!
//! The QueryPlanner classifies query intent and coordinates retrieval across
//! semantic, temporal, causal, and entity dimensions with adaptive weighting.

use crate::{
    error::Result,
    graph::GraphManager,
    index::{TemporalIndex, VectorIndex},
    query::{
        fusion::{FusionEngine, FusedResult},
        intent::{IntentClassification, IntentClassifier},
    },
    storage::StorageEngine,
    types::{MemoryId, Timestamp},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Query planner that coordinates multi-dimensional retrieval
pub struct QueryPlanner {
    storage: Arc<StorageEngine>,
    vector_index: Arc<RwLock<VectorIndex>>,
    temporal_index: Arc<TemporalIndex>,
    graph_manager: Arc<RwLock<GraphManager>>,
    intent_classifier: IntentClassifier,
    fusion_engine: FusionEngine,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(
        storage: Arc<StorageEngine>,
        vector_index: Arc<RwLock<VectorIndex>>,
        temporal_index: Arc<TemporalIndex>,
        graph_manager: Arc<RwLock<GraphManager>>,
    ) -> Self {
        Self {
            storage,
            vector_index,
            temporal_index,
            graph_manager,
            intent_classifier: IntentClassifier::new(),
            fusion_engine: FusionEngine::new(),
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
    ) -> Result<(IntentClassification, Vec<FusedResult>)> {
        // Step 1: Classify intent
        let intent = self.intent_classifier.classify(query_text);

        // Step 2: Retrieve from all dimensions (fetch more to account for filtering)
        let fetch_multiplier = if namespace.is_some() { 5 } else { 3 };
        let mut semantic_scores = self.semantic_search(query_embedding, limit * fetch_multiplier)?;
        let mut temporal_scores = self.temporal_search(limit * fetch_multiplier)?;
        let causal_scores = HashMap::new(); // TODO: Implement causal search
        let mut entity_scores = self.entity_search(query_text, limit * fetch_multiplier)?;

        // Step 2.5: Filter by namespace if provided
        if let Some(ns) = namespace {
            self.filter_by_namespace(&mut semantic_scores, ns)?;
            self.filter_by_namespace(&mut temporal_scores, ns)?;
            self.filter_by_namespace(&mut entity_scores, ns)?;
        }

        // Step 3: Fuse results with adaptive weights
        let mut fused_results = self.fusion_engine.fuse(
            intent.intent,
            &semantic_scores,
            &temporal_scores,
            &causal_scores,
            &entity_scores,
        );

        // Step 4: Limit results
        fused_results.truncate(limit);

        Ok((intent, fused_results))
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

    /// Perform temporal search (recent memories)
    fn temporal_search(&self, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let results = self.temporal_index.recent(limit)?;

        // Calculate temporal scores based on recency
        // Most recent = 1.0, oldest in results = 0.0
        let mut scores = HashMap::new();
        let count = results.len();

        for (i, result) in results.into_iter().enumerate() {
            // Linear decay from 1.0 to 0.0
            let score = if count > 1 {
                1.0 - (i as f32 / (count - 1) as f32)
            } else {
                1.0
            };
            scores.insert(result.id, score);
        }

        Ok(scores)
    }

    /// Perform entity-based search
    fn entity_search(&self, query_text: &str, limit: usize) -> Result<HashMap<MemoryId, f32>> {
        let mut scores = HashMap::new();

        // Extract potential entity names from query (simple capitalized word extraction)
        let potential_entities: Vec<&str> = query_text
            .split_whitespace()
            .filter(|word| {
                word.chars().next().map_or(false, |c| c.is_uppercase())
                    && word.len() > 1
            })
            .collect();

        // For each potential entity, find related memories
        for entity_name in potential_entities {
            if let Ok(Some(entity)) = self.storage.find_entity_by_name(entity_name) {
                // Get memories that mention this entity
                let graph = self.graph_manager.read().unwrap();
                let query_result = graph.get_entity_memories(&entity.id);

                // Score based on entity mention count (popularity)
                let base_score = (entity.mention_count as f32).min(10.0) / 10.0;

                for memory_id in &query_result.memories {
                    *scores.entry(memory_id.clone()).or_insert(0.0) += base_score;
                }
            }
        }

        // Normalize scores
        FusionEngine::normalize_scores(&mut scores);

        // Take top results
        let mut score_vec: Vec<_> = scores.into_iter().collect();
        score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        score_vec.truncate(limit);

        Ok(score_vec.into_iter().collect())
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
        let fetch_limit = if namespace.is_some() { limit * 3 } else { limit };
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
                temporal_score,
                causal_score: 0.0,
                entity_score: 0.0,
                fused_score: temporal_score,
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

        let temporal_index = Arc::new(TemporalIndex::new(Arc::clone(&storage)));
        let graph_manager = Arc::new(RwLock::new(GraphManager::new()));

        let planner = QueryPlanner::new(
            storage,
            vector_index,
            temporal_index,
            graph_manager,
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
        planner.temporal_index.add(&mem1.id, mem1.created_at).unwrap();
        planner.temporal_index.add(&mem2.id, mem2.created_at).unwrap();

        // Search
        let scores = planner.temporal_search(10).unwrap();
        assert_eq!(scores.len(), 2);

        // Most recent should have higher score
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
        planner.temporal_index.add(&mem_id, memory.created_at).unwrap();

        // Execute query
        let (intent, results) = planner
            .query("test query", &vec![0.1; 384], 10, None)
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
        planner.temporal_index.add(&mem1.id, mem1.created_at).unwrap();

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
        planner.temporal_index.add(&mem1_id, mem1.created_at).unwrap();
        planner.temporal_index.add(&mem2_id, mem2.created_at).unwrap();
        planner.temporal_index.add(&mem3_id, mem3.created_at).unwrap();

        // Query with ns1 filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns1"))
            .unwrap();

        // Should only contain mem1
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id == mem1_id));

        // Query with ns2 filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some("ns2"))
            .unwrap();

        // Should only contain mem2
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.id == mem2_id));

        // Query with default namespace filter
        let (_, results) = planner
            .query("test", &vec![0.1; 384], 10, Some(""))
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
        let mut mem1 = Memory::new_with_timestamp(
            "NS1 memory".to_string(),
            vec![0.1; 384],
            ts1,
        );
        mem1.set_namespace("ns1");
        let mem1_id = mem1.id.clone();

        let mut mem2 = Memory::new_with_timestamp(
            "NS2 memory".to_string(),
            vec![0.2; 384],
            ts2,
        );
        mem2.set_namespace("ns2");
        let mem2_id = mem2.id.clone();

        // Store and index
        planner.storage.store_memory(&mem1).unwrap();
        planner.storage.store_memory(&mem2).unwrap();
        planner.temporal_index.add(&mem1_id, mem1.created_at).unwrap();
        planner.temporal_index.add(&mem2_id, mem2.created_at).unwrap();

        // First verify memories are stored correctly
        let retrieved_mem1 = planner.storage.get_memory(&mem1_id).unwrap().unwrap();
        assert_eq!(retrieved_mem1.get_namespace(), "ns1");

        // Query without namespace filter first
        let all_results = planner
            .temporal_range_query(now.subtract_days(2), now, 10, None)
            .unwrap();
        assert_eq!(all_results.len(), 2, "Should find both memories without filter");

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
}
