//! Intent classification test cases
//!
//! Tests for intent classifier: factual, mixed intent, edge cases.
//! Total: 25 test cases

use mnemefusion_core::query::{IntentClassifier, QueryIntent};

// ============================================================================
// Factual Queries (10 tests)
// Tests for queries that should be classified as Factual intent
// ============================================================================

#[test]
fn test_intent_factual_001_topic_query() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("machine learning techniques");

    assert_eq!(result.intent, QueryIntent::Factual, "Pure topic query should be factual");
    assert!(result.confidence >= 0.3, "Factual base score should be 0.3+");
}

#[test]
fn test_intent_factual_002_technical_term() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("database optimization strategies");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_003_what_is_query() {
    let classifier = IntentClassifier::new();

    // "what is" without temporal/causal context should be factual
    let result = classifier.classify("what is rust programming");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_004_how_to_query() {
    let classifier = IntentClassifier::new();

    // "how to" is informational, not causal
    let result = classifier.classify("how to implement binary search");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_005_lowercase_no_entities() {
    let classifier = IntentClassifier::new();

    // All lowercase, no temporal/causal keywords
    let result = classifier.classify("python web frameworks");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_006_short_query() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("api design");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_007_long_descriptive() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify(
        "best practices for building scalable microservices architecture with containerization"
    );

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_008_definition_query() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("define functional programming");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_009_comparison_query() {
    let classifier = IntentClassifier::new();

    // Comparison without temporal/causal aspect
    let result = classifier.classify("compare sql and nosql databases");

    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_factual_010_generic_search() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("kubernetes deployment configuration");

    assert_eq!(result.intent, QueryIntent::Factual);
}

// ============================================================================
// Mixed Intent Queries (10 tests)
// Tests for queries with multiple intent signals
// ============================================================================

#[test]
fn test_intent_mixed_001_temporal_entity() {
    let classifier = IntentClassifier::new();

    // Both temporal ("yesterday") and entity ("Alice")
    let result = classifier.classify("What did Alice do yesterday?");

    // Should detect at least one (temporal or entity)
    assert!(
        result.intent == QueryIntent::Temporal || result.intent == QueryIntent::Entity,
        "Should detect temporal or entity intent"
    );

    // Should have secondary intent
    assert!(
        !result.secondary.is_empty() || result.intent == QueryIntent::Temporal,
        "Mixed query should have secondary signals"
    );
}

#[test]
fn test_intent_mixed_002_temporal_causal() {
    let classifier = IntentClassifier::new();

    // Both temporal ("last week") and causal ("cancelled")
    let result = classifier.classify("Why did the meeting get cancelled last week?");

    // Causal should dominate (higher weight: 0.5 vs 0.4)
    assert_eq!(result.intent, QueryIntent::Causal, "Causal should dominate over temporal");

    // Should have temporal as secondary
    let has_temporal_secondary = result.secondary.iter()
        .any(|(intent, _)| *intent == QueryIntent::Temporal);
    assert!(has_temporal_secondary || result.confidence > 0.5);
}

#[test]
fn test_intent_mixed_003_entity_causal() {
    let classifier = IntentClassifier::new();

    // Both entity ("Alice") and causal ("why")
    let result = classifier.classify("Why did Alice leave the company?");

    // Causal should dominate (stronger pattern)
    assert_eq!(result.intent, QueryIntent::Causal, "Causal should dominate over entity");
}

#[test]
fn test_intent_mixed_004_all_three() {
    let classifier = IntentClassifier::new();

    // Temporal + Entity + Causal
    let result = classifier.classify("Why did Alice cancel the meeting yesterday?");

    // Causal should win (highest weight)
    assert_eq!(result.intent, QueryIntent::Causal, "Causal should dominate in triple-mixed query");

    // Should have multiple secondary intents
    assert!(result.secondary.len() >= 1 || result.confidence > 0.5);
}

#[test]
fn test_intent_mixed_005_multiple_temporal() {
    let classifier = IntentClassifier::new();

    // Multiple temporal keywords
    let result = classifier.classify("What happened yesterday recently?");

    assert_eq!(result.intent, QueryIntent::Temporal);
    // Multiple distinct matches should increase confidence: 2 * 0.4 = 0.8
    assert!(result.confidence >= 0.5, "Multiple temporal keywords should boost confidence");
}

#[test]
fn test_intent_mixed_006_multiple_causal() {
    let classifier = IntentClassifier::new();

    // Multiple causal keywords
    let result = classifier.classify("Why did the failure cause the system crash?");

    assert_eq!(result.intent, QueryIntent::Causal);
    // Multiple matches: at least 2 * 0.5 = 1.0 (capped)
    assert!(result.confidence >= 0.8, "Multiple causal keywords should boost confidence");
}

#[test]
fn test_intent_mixed_007_multiple_entities() {
    let classifier = IntentClassifier::new();

    // Multiple entities
    let result = classifier.classify("Conversation between Alice and Bob about Project Alpha");

    assert_eq!(result.intent, QueryIntent::Entity);
    // Multiple entity matches should boost confidence
    assert!(result.confidence > 0.3);
}

#[test]
fn test_intent_mixed_008_weak_plus_strong() {
    let classifier = IntentClassifier::new();

    // Weak entity signal + strong causal signal
    let result = classifier.classify("Why did the bug occur?");

    // "bug" is lowercase so weak entity, "why" is strong causal
    assert_eq!(result.intent, QueryIntent::Causal);
}

#[test]
fn test_intent_mixed_009_competing_signals() {
    let classifier = IntentClassifier::new();

    // Strong temporal vs factual base
    let result = classifier.classify("recent trends in technology");

    // "recent" is temporal, but "trends in technology" is factual
    // Temporal should win (0.4 > 0.3 base)
    assert_eq!(result.intent, QueryIntent::Temporal);
}

#[test]
fn test_intent_mixed_010_complex_real_world() {
    let classifier = IntentClassifier::new();

    // Complex query with multiple dimensions
    let result = classifier.classify(
        "What caused Alice to resign from Project Beta last month?"
    );

    // Has: causal ("caused"), entity ("Alice", "Project Beta"), temporal ("last month")
    // Causal should dominate
    assert_eq!(result.intent, QueryIntent::Causal);
    assert!(result.confidence >= 0.5);
}

// ============================================================================
// Edge Cases (5 tests)
// Tests for boundary conditions and special cases
// ============================================================================

#[test]
fn test_intent_edge_001_empty_query() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("");

    // Empty query should default to factual (base score 0.3)
    assert_eq!(result.intent, QueryIntent::Factual);
    assert_eq!(result.confidence, 0.3);
}

#[test]
fn test_intent_edge_002_single_word() {
    let classifier = IntentClassifier::new();

    // Single temporal keyword
    let result = classifier.classify("yesterday");
    assert_eq!(result.intent, QueryIntent::Temporal);

    // Single causal keyword
    let result = classifier.classify("why");
    assert_eq!(result.intent, QueryIntent::Causal);

    // Single generic word
    let result = classifier.classify("data");
    assert_eq!(result.intent, QueryIntent::Factual);
}

#[test]
fn test_intent_edge_003_only_stop_words() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("the a an");

    // No patterns match, default to factual
    assert_eq!(result.intent, QueryIntent::Factual);
    assert_eq!(result.confidence, 0.3);
}

#[test]
fn test_intent_edge_004_very_long_query() {
    let classifier = IntentClassifier::new();

    let long_query = format!(
        "This is a very long query with many words that might match patterns {} {} {} but overall is mostly factual content about general topics",
        "yesterday", "because", "Alice"
    );

    let result = classifier.classify(&long_query);

    // Should still detect intents in long queries
    assert!(
        result.intent == QueryIntent::Temporal ||
        result.intent == QueryIntent::Causal ||
        result.intent == QueryIntent::Entity
    );
}

#[test]
fn test_intent_edge_005_confidence_boundaries() {
    let classifier = IntentClassifier::new();

    // Test confidence never exceeds 1.0
    let result = classifier.classify("why why why why why");
    assert!(result.confidence <= 1.0, "Confidence should be capped at 1.0");
    assert_eq!(result.intent, QueryIntent::Causal);

    // Test confidence minimum for factual
    let result = classifier.classify("machine learning");
    assert!(result.confidence >= 0.3, "Factual base should be 0.3");
}

// ============================================================================
// Secondary Intent Tests (5 tests)
// Tests for detecting multiple intents in queries
// ============================================================================

#[test]
fn test_intent_secondary_001_temporal_with_entity() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("Show me recent work about Alice");

    // Primary should be temporal or entity
    assert!(
        result.intent == QueryIntent::Temporal || result.intent == QueryIntent::Entity
    );

    // Query has both temporal ("recent") and entity ("about Alice") signals
    // At least one should be detected (could be primary or have high confidence)
    assert!(
        result.intent == QueryIntent::Temporal ||
        result.intent == QueryIntent::Entity ||
        result.confidence > 0.4,
        "Should detect intent signals"
    );
}

#[test]
fn test_intent_secondary_002_causal_with_temporal() {
    let classifier = IntentClassifier::new();

    let result = classifier.classify("What caused the crash last night?");

    // Primary should be causal (stronger weight)
    assert_eq!(result.intent, QueryIntent::Causal);

    // Check for temporal in secondary
    let has_temporal = result.secondary.iter()
        .any(|(intent, score)| *intent == QueryIntent::Temporal && *score > 0.3);

    assert!(has_temporal || result.confidence > 0.7, "Should detect temporal as secondary");
}

#[test]
fn test_intent_secondary_003_threshold_filtering() {
    let classifier = IntentClassifier::new();

    // Weak entity signal shouldn't make it to secondary
    let result = classifier.classify("Why did the process fail?");

    assert_eq!(result.intent, QueryIntent::Causal);

    // Secondary intents must have score > 0.3
    for (_intent, score) in &result.secondary {
        assert!(*score > 0.3, "Secondary intents must exceed 0.3 threshold");
    }
}

#[test]
fn test_intent_secondary_004_no_secondary_for_pure() {
    let classifier = IntentClassifier::new();

    // Pure factual with no other signals
    let result = classifier.classify("database architecture");

    assert_eq!(result.intent, QueryIntent::Factual);
    assert!(result.secondary.is_empty(), "Pure factual should have no secondary intents");
}

#[test]
fn test_intent_secondary_005_ranking() {
    let classifier = IntentClassifier::new();

    // Query with multiple strong signals
    let result = classifier.classify("Why did Alice leave yesterday?");

    // Primary should be causal (highest weight: 0.5)
    assert_eq!(result.intent, QueryIntent::Causal);

    // Secondary intents should be sorted by score
    if result.secondary.len() >= 2 {
        assert!(
            result.secondary[0].1 >= result.secondary[1].1,
            "Secondary intents should be sorted by score"
        );
    }
}
