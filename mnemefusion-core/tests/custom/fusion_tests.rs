//! Adaptive fusion test cases
//!
//! Tests for adaptive weight fusion: fusion improvement, weight validation, normalization.
//! Total: 10 test cases

use super::test_utils::*;
use mnemefusion_core::{Config, QueryIntent};
use std::collections::HashMap;

// Helper to get timestamp N days ago
fn days_ago(days: u64) -> mnemefusion_core::types::Timestamp {
    mnemefusion_core::types::Timestamp::now().subtract_days(days)
}

// ============================================================================
// Fusion Improvement Tests (5 tests)
// Tests showing that fusion beats semantic-only retrieval
// ============================================================================

#[test]
fn test_fusion_improvement_001_temporal_ranking() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories with varying recency
    // Include temporal expressions in content for content-based temporal matching
    let old_relevant = ctx.add_memory(&TestMemory {
        id: None,
        content: "Important project meeting discussion from last month".to_string(),
        embedding: generate_test_embedding("meeting project month", 384),
        timestamp: Some(days_ago(30)),
        metadata: HashMap::new(),
    });

    let recent_relevant = ctx.add_memory(&TestMemory {
        id: None,
        content: "Project meeting notes from yesterday".to_string(),
        embedding: generate_test_embedding("meeting project notes yesterday", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    let recent_irrelevant = ctx.add_memory(&TestMemory {
        id: None,
        content: "Random unrelated content".to_string(),
        embedding: generate_test_embedding("random content", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    // Query with temporal intent
    let query_emb = generate_test_embedding("recent project meetings", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What happened in recent project meetings?",
            query_emb.clone(),
            10,
            None,
            None,
        )
        .unwrap();

    // Recent relevant memory should rank higher than old relevant
    // This shows temporal dimension helping
    let recent_pos = results.iter().position(|(m, _)| m.id == recent_relevant);
    let old_pos = results.iter().position(|(m, _)| m.id == old_relevant);

    assert!(recent_pos.is_some() && old_pos.is_some());
    assert!(
        recent_pos.unwrap() < old_pos.unwrap(),
        "Recent relevant memory should rank higher than old relevant (fusion at work)"
    );
}

#[test]
fn test_fusion_improvement_002_causal_ranking() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories with and without causal links
    let cause = ctx.add_memory(&TestMemory {
        id: None,
        content: "Server load increased dramatically".to_string(),
        embedding: generate_test_embedding("server load increase", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let effect = ctx.add_memory(&TestMemory {
        id: None,
        content: "Database timeout errors occurred".to_string(),
        embedding: generate_test_embedding("database timeout errors", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let unrelated = ctx.add_memory(&TestMemory {
        id: None,
        content: "Database backup completed successfully".to_string(),
        embedding: generate_test_embedding("database backup success", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Add causal link
    ctx.add_causal_link(&CausalLink {
        from_content: "Server load increased dramatically".to_string(),
        to_content: "Database timeout errors occurred".to_string(),
        confidence: 0.9,
        evidence: "High load causes timeouts".to_string(),
    });

    // Causal query
    let query_emb = generate_test_embedding("database timeout why", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "Why did database timeouts happen?",
            query_emb.clone(),
            10,
            None,
            None,
        )
        .unwrap();

    // Should detect causal intent
    assert_eq!(
        intent.intent,
        QueryIntent::Causal,
        "Should detect causal intent"
    );

    // Should return results including the causally related memory
    let has_cause = results.iter().any(|(m, _)| m.id == cause);
    let has_effect = results.iter().any(|(m, _)| m.id == effect);

    assert!(
        has_cause || has_effect,
        "Should return causally related memories in results"
    );
}

#[test]
fn test_fusion_improvement_003_entity_filtering() {
    let mut ctx = TestContext::new(Config::default());

    // Add memories mentioning different entities
    let alice_memory = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice completed the code review".to_string(),
        embedding: generate_test_embedding("completed code review", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    let bob_memory = ctx.add_memory(&TestMemory {
        id: None,
        content: "Bob completed the code review".to_string(),
        embedding: generate_test_embedding("completed code review", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Entity query
    let query_emb = generate_test_embedding("code review completed", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query("Show me Alice's code reviews", query_emb.clone(), 10, None, None)
        .unwrap();

    // Should return results (entity dimension helps filter/rank)
    assert!(
        !results.is_empty(),
        "Should return results for entity query"
    );

    // Alice's memory should be present
    let has_alice = results.iter().any(|(m, _)| m.id == alice_memory);
    assert!(has_alice, "Alice's memory should be in results");
}

#[test]
fn test_fusion_improvement_004_multi_dimensional_boost() {
    let mut ctx = TestContext::new(Config::default());

    // Perfect match: semantic + temporal + entity
    let perfect = ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice fixed the critical bug".to_string(),
        embedding: generate_test_embedding("Alice bug fix critical", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    // Semantic only
    let semantic_only = ctx.add_memory(&TestMemory {
        id: None,
        content: "Bug fix was critical".to_string(),
        embedding: generate_test_embedding("bug fix critical important", 384),
        timestamp: Some(days_ago(30)),
        metadata: HashMap::new(),
    });

    // Query matching multiple dimensions
    let query_emb = generate_test_embedding("Alice bug fix critical", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "What did Alice do recently about the critical bug?",
            query_emb.clone(),
            10,
            None,
            None,
        )
        .unwrap();

    // Multi-dimensional match should rank higher
    let perfect_pos = results.iter().position(|(m, _)| m.id == perfect);
    let semantic_pos = results.iter().position(|(m, _)| m.id == semantic_only);

    if perfect_pos.is_some() && semantic_pos.is_some() {
        assert!(
            perfect_pos.unwrap() <= semantic_pos.unwrap(),
            "Multi-dimensional match should rank as high or higher"
        );
    }
}

#[test]
fn test_fusion_improvement_005_intent_adaptation() {
    let mut ctx = TestContext::new(Config::default());

    // Add temporal-relevant memory
    let recent = ctx.add_memory(&TestMemory {
        id: None,
        content: "Generic content here".to_string(),
        embedding: generate_test_embedding("generic content", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    let old = ctx.add_memory(&TestMemory {
        id: None,
        content: "Very similar generic content".to_string(),
        embedding: generate_test_embedding("generic content similar", 384),
        timestamp: Some(days_ago(30)),
        metadata: HashMap::new(),
    });

    // Temporal query should rank recent higher
    let query_emb = generate_test_embedding("generic content", 384);
    let (_intent, results, _profile_ctx) = ctx
        .engine
        .query("Show me recent generic content", query_emb.clone(), 10, None, None)
        .unwrap();

    let recent_pos = results.iter().position(|(m, _)| m.id == recent);
    let old_pos = results.iter().position(|(m, _)| m.id == old);

    if recent_pos.is_some() && old_pos.is_some() {
        assert!(
            recent_pos.unwrap() < old_pos.unwrap(),
            "Intent-adapted weights should favor recent memory"
        );
    }
}

// ============================================================================
// Weight Validation Tests (5 tests)
// Tests verifying that weights adapt correctly to query intent
// ============================================================================

#[test]
fn test_fusion_weights_001_temporal_query() {
    let mut ctx = TestContext::new(Config::default());

    // Add a memory
    ctx.add_memory(&TestMemory {
        id: None,
        content: "Test memory content".to_string(),
        embedding: generate_test_embedding("test memory", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    // Temporal query
    let query_emb = generate_test_embedding("test memory", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("What happened yesterday?", query_emb.clone(), 10, None, None)
        .unwrap();

    // Should detect temporal intent
    assert_eq!(intent.intent, QueryIntent::Temporal);

    // Note: Weights are internal to fusion engine and not exposed directly
    // We validate intent detection as proxy for weight adaptation
    assert!(
        intent.confidence > 0.3,
        "Temporal query should have reasonable confidence"
    );
}

#[test]
fn test_fusion_weights_002_causal_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Something happened here".to_string(),
        embedding: generate_test_embedding("something happened", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Causal query
    let query_emb = generate_test_embedding("something happened", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Why did this happen?", query_emb.clone(), 10, None, None)
        .unwrap();

    // Should detect causal intent
    assert_eq!(intent.intent, QueryIntent::Causal);
    assert!(
        intent.confidence >= 0.4,
        "Causal query should have high confidence"
    );
}

#[test]
fn test_fusion_weights_003_entity_query() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice did something important".to_string(),
        embedding: generate_test_embedding("Alice something important", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Entity query
    let query_emb = generate_test_embedding("Alice work", 384);
    let (intent, _results, _profile_ctx) = ctx
        .engine
        .query("Tell me about Alice's work", query_emb.clone(), 10, None, None)
        .unwrap();

    // Should detect entity intent
    assert_eq!(intent.intent, QueryIntent::Entity);
}

#[test]
fn test_fusion_weights_004_factual_default() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Machine learning is a subset of AI".to_string(),
        embedding: generate_test_embedding("machine learning AI subset", 384),
        timestamp: None,
        metadata: HashMap::new(),
    });

    // Factual query (no special keywords)
    let query_emb = generate_test_embedding("machine learning AI", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "machine learning and artificial intelligence",
            query_emb.clone(),
            10,
            None,
            None,
        )
        .unwrap();

    // Should default to factual
    assert_eq!(intent.intent, QueryIntent::Factual);
    assert!(
        !results.is_empty(),
        "Should return results for factual query"
    );
}

#[test]
fn test_fusion_weights_005_mixed_intent_prioritization() {
    let mut ctx = TestContext::new(Config::default());

    ctx.add_memory(&TestMemory {
        id: None,
        content: "Alice worked on the project".to_string(),
        embedding: generate_test_embedding("Alice project work", 384),
        timestamp: Some(days_ago(1)),
        metadata: HashMap::new(),
    });

    // Mixed query: temporal + entity + causal
    let query_emb = generate_test_embedding("Alice project yesterday why", 384);
    let (intent, results, _profile_ctx) = ctx
        .engine
        .query(
            "Why did Alice work on the project yesterday?",
            query_emb.clone(),
            10,
            None,
            None,
        )
        .unwrap();

    // Causal should dominate (highest weight: 0.5 vs 0.4 temporal, 0.2 entity)
    assert_eq!(
        intent.intent,
        QueryIntent::Causal,
        "Causal should dominate in mixed queries"
    );
    assert!(!results.is_empty(), "Should return results");

    // Should have secondary intents detected
    let has_secondary = !intent.secondary.is_empty();
    assert!(
        has_secondary || intent.confidence > 0.5,
        "Mixed query should show multiple intent signals"
    );
}
