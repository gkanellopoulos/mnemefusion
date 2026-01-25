//! Test utilities for custom test cases
//!
//! Provides helpers for setting up test databases, running queries,
//! and asserting expected results for temporal, causal, entity, and intent tests.

use mnemefusion_core::{
    query::intent::QueryIntent,
    types::{MemoryId, Timestamp},
    Config, MemoryEngine,
};
use std::collections::HashMap;
use tempfile::TempDir;

/// Test case setup configuration
#[derive(Debug, Clone)]
pub struct TestSetup {
    pub memories: Vec<TestMemory>,
    pub causal_links: Vec<CausalLink>,
    pub config: Config,
}

/// A memory for test setup
#[derive(Debug, Clone)]
pub struct TestMemory {
    pub id: Option<String>,
    pub content: String,
    pub embedding: Vec<f32>,
    pub timestamp: Option<Timestamp>,
    pub metadata: HashMap<String, String>,
}

/// A causal link for test setup
#[derive(Debug, Clone)]
pub struct CausalLink {
    pub from_content: String,
    pub to_content: String,
    pub confidence: f32,
    pub evidence: String,
}

/// Test expectations
#[derive(Debug, Clone)]
pub struct TestExpectations {
    pub intent: Option<QueryIntent>,
    pub intent_confidence_min: Option<f32>,
    pub results_count_min: Option<usize>,
    pub results_must_include: Vec<String>,
    pub results_ordered: Vec<String>,
    pub temporal_score_threshold: Option<f32>,
    pub fusion_weights: Option<FusionWeightExpectations>,
    pub causal_chain: Vec<String>,
    pub entities_detected: Vec<String>,
}

/// Fusion weight expectations
#[derive(Debug, Clone)]
pub struct FusionWeightExpectations {
    pub semantic_min: Option<f32>,
    pub semantic_max: Option<f32>,
    pub temporal_min: Option<f32>,
    pub temporal_max: Option<f32>,
    pub causal_min: Option<f32>,
    pub causal_max: Option<f32>,
    pub entity_min: Option<f32>,
    pub entity_max: Option<f32>,
}

/// Test context that holds the database and mappings
pub struct TestContext {
    pub engine: MemoryEngine,
    pub _temp_dir: TempDir,
    pub content_to_id: HashMap<String, MemoryId>,
}

impl TestContext {
    /// Create a new test context with a temporary database
    pub fn new(config: Config) -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("test.mfdb");
        let engine = MemoryEngine::open(&db_path, config).expect("Failed to open test database");

        Self {
            engine,
            _temp_dir: temp_dir,
            content_to_id: HashMap::new(),
        }
    }

    /// Add a test memory to the database
    pub fn add_memory(&mut self, memory: &TestMemory) -> MemoryId {
        let id = self
            .engine
            .add(
                memory.content.clone(),
                memory.embedding.clone(),
                if memory.metadata.is_empty() {
                    None
                } else {
                    Some(memory.metadata.clone())
                },
                memory.timestamp,
                None, // source
                None, // namespace
            )
            .expect("Failed to add memory");

        self.content_to_id
            .insert(memory.content.clone(), id.clone());
        id
    }

    /// Add a causal link between two memories
    pub fn add_causal_link(&mut self, link: &CausalLink) {
        let from_id = self
            .content_to_id
            .get(&link.from_content)
            .expect(&format!("Memory not found: {}", link.from_content));
        let to_id = self
            .content_to_id
            .get(&link.to_content)
            .expect(&format!("Memory not found: {}", link.to_content));

        self.engine
            .add_causal_link(from_id, to_id, link.confidence, link.evidence.clone())
            .expect("Failed to add causal link");
    }

    /// Get memory ID by content
    pub fn get_id_by_content(&self, content: &str) -> Option<&MemoryId> {
        self.content_to_id.get(content)
    }
}

/// Generate a simple test embedding (for testing purposes only)
///
/// Creates a deterministic embedding based on the content hash.
/// For real tests, use pre-computed embeddings from a model.
pub fn generate_test_embedding(content: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dim];

    // Simple hash-based generation for determinism
    let hash = content
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

    for i in 0..dim {
        let seed = hash.wrapping_add(i as u64);
        embedding[i] = ((seed % 1000) as f32) / 1000.0;
    }

    // Normalize to unit vector
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in &mut embedding {
            *x /= magnitude;
        }
    }

    embedding
}

/// Assert that the intent matches expectations
pub fn assert_intent(actual: QueryIntent, expected: QueryIntent, test_name: &str) {
    assert_eq!(
        actual, expected,
        "{}: Expected intent {:?}, got {:?}",
        test_name, expected, actual
    );
}

/// Assert that intent confidence meets minimum threshold
pub fn assert_intent_confidence(actual: f32, min_expected: f32, test_name: &str) {
    assert!(
        actual >= min_expected,
        "{}: Expected confidence >= {}, got {}",
        test_name,
        min_expected,
        actual
    );
}

/// Assert that results include expected memories
pub fn assert_results_include(result_contents: &[String], expected: &[String], test_name: &str) {
    for expected_content in expected {
        assert!(
            result_contents.contains(expected_content),
            "{}: Expected result '{}' not found in results: {:?}",
            test_name,
            expected_content,
            result_contents
        );
    }
}

/// Assert that results are in expected order
pub fn assert_results_ordered(
    actual_ids: &[MemoryId],
    expected_contents: &[String],
    content_to_id: &HashMap<String, MemoryId>,
    test_name: &str,
) {
    let expected_ids: Vec<&MemoryId> = expected_contents
        .iter()
        .map(|content| {
            content_to_id
                .get(content)
                .expect(&format!("Content not found: {}", content))
        })
        .collect();

    assert_eq!(
        actual_ids.len(),
        expected_ids.len(),
        "{}: Result count mismatch. Expected {}, got {}",
        test_name,
        expected_ids.len(),
        actual_ids.len()
    );

    for (i, (actual, expected)) in actual_ids.iter().zip(expected_ids.iter()).enumerate() {
        assert_eq!(
            actual, *expected,
            "{}: Result order mismatch at position {}. Expected {:?}, got {:?}",
            test_name, i, expected, actual
        );
    }
}

/// Assert that fusion weights are within expected ranges
pub fn assert_fusion_weights(
    semantic: f32,
    temporal: f32,
    causal: f32,
    entity: f32,
    expectations: &FusionWeightExpectations,
    test_name: &str,
) {
    // Weights should sum to approximately 1.0
    let sum = semantic + temporal + causal + entity;
    assert!(
        (sum - 1.0).abs() < 0.01,
        "{}: Fusion weights should sum to 1.0, got {}",
        test_name,
        sum
    );

    // Check individual weight ranges
    if let Some(min) = expectations.semantic_min {
        assert!(
            semantic >= min,
            "{}: semantic weight {} < min {}",
            test_name,
            semantic,
            min
        );
    }
    if let Some(max) = expectations.semantic_max {
        assert!(
            semantic <= max,
            "{}: semantic weight {} > max {}",
            test_name,
            semantic,
            max
        );
    }
    if let Some(min) = expectations.temporal_min {
        assert!(
            temporal >= min,
            "{}: temporal weight {} < min {}",
            test_name,
            temporal,
            min
        );
    }
    if let Some(max) = expectations.temporal_max {
        assert!(
            temporal <= max,
            "{}: temporal weight {} > max {}",
            test_name,
            temporal,
            max
        );
    }
    if let Some(min) = expectations.causal_min {
        assert!(
            causal >= min,
            "{}: causal weight {} < min {}",
            test_name,
            causal,
            min
        );
    }
    if let Some(max) = expectations.causal_max {
        assert!(
            causal <= max,
            "{}: causal weight {} > max {}",
            test_name,
            causal,
            max
        );
    }
    if let Some(min) = expectations.entity_min {
        assert!(
            entity >= min,
            "{}: entity weight {} < min {}",
            test_name,
            entity,
            min
        );
    }
    if let Some(max) = expectations.entity_max {
        assert!(
            entity <= max,
            "{}: entity weight {} > max {}",
            test_name,
            entity,
            max
        );
    }
}

/// Assert that results are within a time range
pub fn assert_results_in_timerange(
    result_ids: &[MemoryId],
    start: Timestamp,
    end: Timestamp,
    engine: &MemoryEngine,
    test_name: &str,
) {
    for id in result_ids {
        let memory = engine
            .get(id)
            .expect("Failed to get memory")
            .expect("Memory not found");

        // created_at is already a Timestamp
        assert!(
            memory.created_at >= start && memory.created_at <= end,
            "{}: Memory timestamp {} not in range [{}, {}]",
            test_name,
            memory.created_at,
            start,
            end
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embedding_deterministic() {
        let emb1 = generate_test_embedding("test content", 384);
        let emb2 = generate_test_embedding("test content", 384);
        assert_eq!(emb1, emb2, "Embeddings should be deterministic");
    }

    #[test]
    fn test_generate_embedding_normalized() {
        let emb = generate_test_embedding("test", 384);
        let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.01,
            "Embedding should be normalized"
        );
    }

    #[test]
    fn test_context_creation() {
        let config = Config::default();
        let ctx = TestContext::new(config);
        assert!(ctx.content_to_id.is_empty());
    }

    #[test]
    fn test_add_memory() {
        let config = Config::default();
        let mut ctx = TestContext::new(config);

        let memory = TestMemory {
            id: None,
            content: "Test content".to_string(),
            embedding: generate_test_embedding("Test content", 384),
            timestamp: None,
            metadata: HashMap::new(),
        };

        let id = ctx.add_memory(&memory);
        assert!(ctx.get_id_by_content("Test content").is_some());

        // Verify we can retrieve it
        let retrieved = ctx.engine.get(&id).unwrap().unwrap();
        assert_eq!(retrieved.content, "Test content");
    }
}
