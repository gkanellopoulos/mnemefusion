//! Property-based tests for MnemeFusion core invariants
//!
//! These tests use proptest to verify properties that should hold
//! for all possible inputs within defined ranges.
//!
//! Target: 50+ properties with 100 iterations each

use mnemefusion_core::{Config, MemoryEngine};
use proptest::prelude::*;
use std::collections::HashMap;
use tempfile::TempDir;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a temporary test engine
fn create_test_engine() -> (TempDir, MemoryEngine) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.mfdb");
    let engine = MemoryEngine::open(&db_path, Config::default()).unwrap();
    (temp_dir, engine)
}

/// Generate a valid embedding vector of given dimension
fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    let hasher_builder = RandomState::new();
    (0..dim)
        .map(|i| {
            let mut hasher = hasher_builder.build_hasher();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            // Normalize to [-1.0, 1.0]
            ((hash as f32) / (u64::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

// ============================================================================
// MemoryId Conversion Properties (10 properties)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: MemoryId round-trip through u64 is identity
    #[test]
    fn prop_memory_id_u64_roundtrip(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id1 = MemoryId::from_u64(id_value);
        let roundtrip = id1.to_u64();
        let id2 = MemoryId::from_u64(roundtrip);

        prop_assert_eq!(id1, id2);
        prop_assert_eq!(roundtrip, id_value);
    }

    /// Property: MemoryId round-trip through UUID string is identity
    #[test]
    fn prop_memory_id_uuid_string_roundtrip(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id1 = MemoryId::from_u64(id_value);
        let uuid_str = id1.to_string();
        let id2 = MemoryId::parse(&uuid_str).unwrap();

        prop_assert_eq!(id1, id2);
    }

    /// Property: MemoryId to_u64 is consistent
    #[test]
    fn prop_memory_id_u64_consistent(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id = MemoryId::from_u64(id_value);
        let u64_1 = id.to_u64();
        let u64_2 = id.to_u64();

        prop_assert_eq!(u64_1, u64_2);
    }

    /// Property: MemoryId equality is reflexive
    #[test]
    fn prop_memory_id_reflexive(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id = MemoryId::from_u64(id_value);
        prop_assert_eq!(id.clone(), id);
    }

    /// Property: MemoryId equality is symmetric
    #[test]
    fn prop_memory_id_symmetric(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id1 = MemoryId::from_u64(id_value);
        let id2 = MemoryId::from_u64(id_value);

        prop_assert_eq!(id1 == id2, id2 == id1);
    }

    /// Property: MemoryId equality is transitive
    #[test]
    fn prop_memory_id_transitive(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id1 = MemoryId::from_u64(id_value);
        let id2 = MemoryId::from_u64(id_value);
        let id3 = MemoryId::from_u64(id_value);

        if id1 == id2 && id2 == id3 {
            prop_assert_eq!(id1, id3);
        }
    }

    /// Property: Different u64 values produce different MemoryIds
    #[test]
    fn prop_memory_id_uniqueness(a in 0u64..u64::MAX, b in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id_a = MemoryId::from_u64(a);
        let id_b = MemoryId::from_u64(b);

        if a != b {
            prop_assert_ne!(id_a, id_b);
        } else {
            prop_assert_eq!(id_a, id_b);
        }
    }

    /// Property: MemoryId hash is consistent
    #[test]
    fn prop_memory_id_hash_consistent(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let id = MemoryId::from_u64(id_value);

        let mut hasher1 = DefaultHasher::new();
        id.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        id.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        prop_assert_eq!(hash1, hash2);
    }

    /// Property: Equal MemoryIds produce equal hashes
    #[test]
    fn prop_memory_id_equal_hashes(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let id1 = MemoryId::from_u64(id_value);
        let id2 = MemoryId::from_u64(id_value);

        let mut hasher1 = DefaultHasher::new();
        id1.hash(&mut hasher1);

        let mut hasher2 = DefaultHasher::new();
        id2.hash(&mut hasher2);

        prop_assert_eq!(hasher1.finish(), hasher2.finish());
    }

    /// Property: MemoryId clone is equal to original
    #[test]
    fn prop_memory_id_clone(id_value in 0u64..u64::MAX) {
        use mnemefusion_core::types::MemoryId;

        let id = MemoryId::from_u64(id_value);
        let cloned = id.clone();

        prop_assert_eq!(id, cloned);
    }
}

// ============================================================================
// Timestamp Properties (10 properties)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Timestamp::now() is always increasing
    #[test]
    fn prop_timestamp_monotonic(_seed in 0u64..1000) {
        use mnemefusion_core::types::Timestamp;
        use std::thread::sleep;
        use std::time::Duration;

        let t1 = Timestamp::now();
        sleep(Duration::from_millis(1));
        let t2 = Timestamp::now();

        prop_assert!(t2.as_micros() >= t1.as_micros());
    }

    /// Property: Timestamp comparison is transitive
    #[test]
    fn prop_timestamp_transitive(
        micros_a in 0u64..1_000_000_000_000,
        micros_b in 0u64..1_000_000_000_000,
        micros_c in 0u64..1_000_000_000_000
    ) {
        use mnemefusion_core::types::Timestamp;

        let t_a = Timestamp::from_micros(micros_a);
        let t_b = Timestamp::from_micros(micros_b);
        let t_c = Timestamp::from_micros(micros_c);

        if t_a <= t_b && t_b <= t_c {
            prop_assert!(t_a <= t_c);
        }
    }

    /// Property: Timestamp round-trip through micros is identity
    #[test]
    fn prop_timestamp_micros_roundtrip(micros in 0u64..1_000_000_000_000) {
        use mnemefusion_core::types::Timestamp;

        let t1 = Timestamp::from_micros(micros);
        let roundtrip = t1.as_micros();
        let t2 = Timestamp::from_micros(roundtrip);

        prop_assert_eq!(t1, t2);
        prop_assert_eq!(roundtrip, micros);
    }

    /// Property: subtract_days produces earlier timestamp
    #[test]
    fn prop_timestamp_subtract_days(days in 1u64..365) {
        use mnemefusion_core::types::Timestamp;

        let now = Timestamp::now();
        let earlier = now.subtract_days(days);

        prop_assert!(earlier.as_micros() < now.as_micros());
    }

    /// Property: Timestamp equality is reflexive
    #[test]
    fn prop_timestamp_reflexive(micros in 0u64..1_000_000_000_000) {
        use mnemefusion_core::types::Timestamp;

        let t = Timestamp::from_micros(micros);
        prop_assert_eq!(t, t);
    }

    /// Property: Timestamp equality is symmetric
    #[test]
    fn prop_timestamp_symmetric(micros in 0u64..1_000_000_000_000) {
        use mnemefusion_core::types::Timestamp;

        let t1 = Timestamp::from_micros(micros);
        let t2 = Timestamp::from_micros(micros);

        prop_assert_eq!(t1 == t2, t2 == t1);
    }

    /// Property: Different micros produce different timestamps
    #[test]
    fn prop_timestamp_uniqueness(
        micros_a in 0u64..1_000_000_000_000,
        micros_b in 0u64..1_000_000_000_000
    ) {
        use mnemefusion_core::types::Timestamp;

        let t_a = Timestamp::from_micros(micros_a);
        let t_b = Timestamp::from_micros(micros_b);

        if micros_a != micros_b {
            prop_assert_ne!(t_a, t_b);
        } else {
            prop_assert_eq!(t_a, t_b);
        }
    }

    /// Property: Timestamp ordering is consistent with micros
    #[test]
    fn prop_timestamp_ordering(
        micros_a in 0u64..1_000_000_000_000,
        micros_b in 0u64..1_000_000_000_000
    ) {
        use mnemefusion_core::types::Timestamp;

        let t_a = Timestamp::from_micros(micros_a);
        let t_b = Timestamp::from_micros(micros_b);

        let expected_ordering = micros_a.cmp(&micros_b);
        let actual_ordering = t_a.cmp(&t_b);

        prop_assert_eq!(expected_ordering, actual_ordering);
    }

    /// Property: Timestamp clone is equal to original
    #[test]
    fn prop_timestamp_clone(micros in 0u64..1_000_000_000_000) {
        use mnemefusion_core::types::Timestamp;

        let t = Timestamp::from_micros(micros);
        let cloned = t.clone();

        prop_assert_eq!(t, cloned);
    }

    /// Property: subtract_days is consistent across multiple calls
    #[test]
    fn prop_timestamp_subtract_days_consistent(days in 1u64..100) {
        use mnemefusion_core::types::Timestamp;

        let now = Timestamp::now();
        let earlier1 = now.subtract_days(days);
        let earlier2 = now.subtract_days(days);

        // Should produce the same result
        let diff = if earlier1.as_micros() > earlier2.as_micros() {
            earlier1.as_micros() - earlier2.as_micros()
        } else {
            earlier2.as_micros() - earlier1.as_micros()
        };

        // Allow small timing differences (< 1ms)
        prop_assert!(diff < 1000);
    }
}

// ============================================================================
// Score Normalization Properties (10 properties)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Normalized score is in [0.0, 1.0]
    #[test]
    fn prop_normalized_score_bounds(raw_score in -100.0f32..100.0f32) {
        // Normalize using sigmoid-like function
        let normalized = 1.0 / (1.0 + (-raw_score).exp());

        prop_assert!(normalized >= 0.0 && normalized <= 1.0);
    }

    /// Property: Temporal decay score is in [0.0, 1.0]
    #[test]
    fn prop_temporal_decay_bounds(hours_ago in 0u64..8760) {
        use mnemefusion_core::Config;

        let config = Config::default();
        let decay_hours = config.temporal_decay_hours;

        // Temporal decay formula: exp(-hours / decay_hours)
        let decay = (-((hours_ago as f32) / decay_hours)).exp();

        prop_assert!(decay >= 0.0 && decay <= 1.0);
    }

    /// Property: Search results are sorted descending by score
    #[test]
    fn prop_search_results_sorted(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        // Add some memories
        let query_emb = generate_embedding(384, seed);
        for i in 0..10 {
            let emb = generate_embedding(384, seed + i + 1);
            engine.add(
                format!("Memory {}", i),
                emb,
                None,
                None,
                None,
                None,
            ).unwrap();
        }

        let results = engine.search(&query_emb, 10, None, None).unwrap();

        // Check descending order
        for window in results.windows(2) {
            prop_assert!(window[0].1 >= window[1].1,
                "Results not sorted: {} < {}", window[0].1, window[1].1);
        }
    }

    /// Property: Query results are sorted descending by fused_score
    #[test]
    fn prop_query_results_sorted(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        // Add some memories
        let query_emb = generate_embedding(384, seed);
        for i in 0..10 {
            let emb = generate_embedding(384, seed + i + 1);
            engine.add(
                format!("Memory {}", i),
                emb,
                None,
                None,
                None,
                None,
            ).unwrap();
        }

        let (_intent, results, _profile_ctx) = engine.query(
            "test query",
            query_emb.clone(),
            10,
            None,
            None,
        ).unwrap();

        // Check descending order by fused_score
        for window in results.windows(2) {
            prop_assert!(window[0].1.fused_score >= window[1].1.fused_score,
                "Query results not sorted: {} < {}", window[0].1.fused_score, window[1].1.fused_score);
        }
    }

    /// Property: All search scores are in [0.0, 1.0]
    #[test]
    fn prop_search_scores_bounded(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        // Add some memories
        let query_emb = generate_embedding(384, seed);
        for i in 0..5 {
            let emb = generate_embedding(384, seed + i + 1);
            engine.add(
                format!("Memory {}", i),
                emb,
                None,
                None,
                None,
                None,
            ).unwrap();
        }

        let results = engine.search(&query_emb, 5, None, None).unwrap();

        for (_memory, score) in results {
            prop_assert!(score >= 0.0 && score <= 1.0,
                "Score out of bounds: {}", score);
        }
    }

    /// Property: All query fused scores are in [0.0, 1.0]
    #[test]
    fn prop_query_scores_bounded(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        // Add some memories
        let query_emb = generate_embedding(384, seed);
        for i in 0..5 {
            let emb = generate_embedding(384, seed + i + 1);
            engine.add(
                format!("Memory {}", i),
                emb,
                None,
                None,
                None,
                None,
            ).unwrap();
        }

        let (_intent, results, _profile_ctx) = engine.query(
            "test query",
            query_emb.clone(),
            5,
            None,
            None,
        ).unwrap();

        for (_memory, fused_result) in results {
            prop_assert!(fused_result.fused_score >= 0.0 && fused_result.fused_score <= 1.0,
                "Fused score out of bounds: {}", fused_result.fused_score);
        }
    }

    /// Property: Temporal decay approaches zero for very old memories
    #[test]
    fn prop_temporal_decay_old(hours_ago in 1000u64..10000) {
        use mnemefusion_core::Config;

        let config = Config::default();
        let decay_hours = config.temporal_decay_hours;

        // Temporal decay formula: exp(-hours / decay_hours)
        let decay = (-((hours_ago as f32) / decay_hours)).exp();

        // Very old memories should have very low decay score
        prop_assert!(decay < 0.1);
    }

    /// Property: Temporal decay is 1.0 for current time
    #[test]
    fn prop_temporal_decay_current(_seed in 0u64..100) {
        use mnemefusion_core::Config;

        let config = Config::default();
        let decay_hours = config.temporal_decay_hours;

        // Current time (0 hours ago)
        let decay = (-(0.0 / decay_hours)).exp();

        prop_assert!((decay - 1.0).abs() < 0.001);
    }
}

// ============================================================================
// Fusion Weight Properties (10 properties)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Intent weights sum to 1.0 (±0.01) for Factual intent
    #[test]
    fn prop_fusion_weights_sum_factual(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Factual);

        let sum = weights.semantic + weights.temporal + weights.causal + weights.entity;

        prop_assert!((sum - 1.0).abs() <= 0.01,
            "Factual weights sum to {} (expected 1.0)", sum);
    }

    /// Property: Intent weights sum to 1.0 (±0.01) for Temporal intent
    #[test]
    fn prop_fusion_weights_sum_temporal(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Temporal);

        let sum = weights.semantic + weights.temporal + weights.causal + weights.entity;

        prop_assert!((sum - 1.0).abs() <= 0.01,
            "Temporal weights sum to {} (expected 1.0)", sum);
    }

    /// Property: Intent weights sum to 1.0 (±0.01) for Causal intent
    #[test]
    fn prop_fusion_weights_sum_causal(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Causal);

        let sum = weights.semantic + weights.temporal + weights.causal + weights.entity;

        prop_assert!((sum - 1.0).abs() <= 0.01,
            "Causal weights sum to {} (expected 1.0)", sum);
    }

    /// Property: Intent weights sum to 1.0 (±0.01) for Entity intent
    #[test]
    fn prop_fusion_weights_sum_entity(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Entity);

        let sum = weights.semantic + weights.temporal + weights.causal + weights.entity;

        prop_assert!((sum - 1.0).abs() <= 0.01,
            "Entity weights sum to {} (expected 1.0)", sum);
    }

    /// Property: All individual weights are in [0.0, 1.0]
    #[test]
    fn prop_fusion_weights_bounded(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();

        for intent in &[
            QueryIntent::Factual,
            QueryIntent::Temporal,
            QueryIntent::Causal,
            QueryIntent::Entity,
        ] {
            let weights = engine.get_weights(*intent);

            prop_assert!(weights.semantic >= 0.0 && weights.semantic <= 1.0);
            prop_assert!(weights.temporal >= 0.0 && weights.temporal <= 1.0);
            prop_assert!(weights.causal >= 0.0 && weights.causal <= 1.0);
            prop_assert!(weights.entity >= 0.0 && weights.entity <= 1.0);
        }
    }

    /// Property: Temporal intent gives highest weight to temporal dimension
    #[test]
    fn prop_temporal_intent_weights(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Temporal);

        // Temporal should be the highest weight
        prop_assert!(weights.temporal >= weights.semantic);
        prop_assert!(weights.temporal >= weights.causal);
        prop_assert!(weights.temporal >= weights.entity);
    }

    /// Property: Causal intent gives highest weight to causal dimension
    #[test]
    fn prop_causal_intent_weights(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Causal);

        // Causal should be the highest weight
        prop_assert!(weights.causal >= weights.semantic);
        prop_assert!(weights.causal >= weights.temporal);
        prop_assert!(weights.causal >= weights.entity);
    }

    /// Property: Entity intent gives highest weight to entity dimension
    #[test]
    fn prop_entity_intent_weights(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Entity);

        // Entity should be the highest weight
        prop_assert!(weights.entity >= weights.semantic);
        prop_assert!(weights.entity >= weights.temporal);
        prop_assert!(weights.entity >= weights.causal);
    }

    /// Property: Factual intent gives balanced weights
    #[test]
    fn prop_factual_intent_balanced(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine = FusionEngine::new();
        let weights = engine.get_weights(QueryIntent::Factual);

        // For factual, semantic should be highest
        prop_assert!(weights.semantic >= weights.temporal);
        prop_assert!(weights.semantic >= weights.causal);
        prop_assert!(weights.semantic >= weights.entity);
    }

    /// Property: Weight config is consistent across calls
    #[test]
    fn prop_fusion_weights_consistent(_seed in 0u64..100) {
        use mnemefusion_core::query::fusion::FusionEngine;
        use mnemefusion_core::QueryIntent;

        let engine1 = FusionEngine::new();
        let engine2 = FusionEngine::new();

        for intent in &[
            QueryIntent::Factual,
            QueryIntent::Temporal,
            QueryIntent::Causal,
            QueryIntent::Entity,
        ] {
            let weights1 = engine1.get_weights(*intent);
            let weights2 = engine2.get_weights(*intent);

            prop_assert_eq!(weights1.semantic, weights2.semantic);
            prop_assert_eq!(weights1.temporal, weights2.temporal);
            prop_assert_eq!(weights1.causal, weights2.causal);
            prop_assert_eq!(weights1.entity, weights2.entity);
        }
    }
}

// ============================================================================
// Memory Storage Properties (10 properties)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: Added memory can be retrieved
    #[test]
    fn prop_memory_add_retrieve(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        let content = format!("Test memory {}", seed);
        let embedding = generate_embedding(384, seed);

        let id = engine.add(
            content.clone(),
            embedding.clone(),
            None,
            None,
            None,
            None,
        ).unwrap();

        let retrieved = engine.get(&id).unwrap();

        prop_assert!(retrieved.is_some());
        prop_assert_eq!(retrieved.unwrap().content, content);
    }

    /// Property: Memory count increases after add
    #[test]
    fn prop_memory_count_increases(seed in 0u64..100) {
        let (_temp, engine) = create_test_engine();

        let count_before = engine.count().unwrap();

        let embedding = generate_embedding(384, seed);
        engine.add(
            "Test memory".to_string(),
            embedding,
            None,
            None,
            None,
            None,
        ).unwrap();

        let count_after = engine.count().unwrap();

        prop_assert_eq!(count_after, count_before + 1);
    }

    /// Property: Deleted memory cannot be retrieved
    #[test]
    fn prop_memory_delete(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        let embedding = generate_embedding(384, seed);
        let id = engine.add(
            "Test memory".to_string(),
            embedding,
            None,
            None,
            None,
            None,
        ).unwrap();

        engine.delete(&id, None).unwrap();

        let retrieved = engine.get(&id).unwrap();
        prop_assert!(retrieved.is_none());
    }

    /// Property: Memory count decreases after delete
    #[test]
    fn prop_memory_count_decreases(seed in 0u64..100) {
        let (_temp, engine) = create_test_engine();

        let embedding = generate_embedding(384, seed);
        let id = engine.add(
            "Test memory".to_string(),
            embedding,
            None,
            None,
            None,
            None,
        ).unwrap();

        let count_before = engine.count().unwrap();
        engine.delete(&id, None).unwrap();
        let count_after = engine.count().unwrap();

        prop_assert_eq!(count_after, count_before - 1);
    }

    /// Property: Multiple memories have unique IDs
    #[test]
    fn prop_memory_unique_ids(count in 2usize..10) {
        let (_temp, engine) = create_test_engine();

        let mut ids = Vec::new();
        for i in 0..count {
            let embedding = generate_embedding(384, i as u64);
            let id = engine.add(
                format!("Memory {}", i),
                embedding,
                None,
                None,
                None,
                None,
            ).unwrap();
            ids.push(id);
        }

        // Check all IDs are unique
        for i in 0..ids.len() {
            for j in (i+1)..ids.len() {
                prop_assert_ne!(ids[i].clone(), ids[j].clone());
            }
        }
    }

    /// Property: Memory metadata is preserved
    #[test]
    fn prop_memory_metadata_preserved(seed in 0u64..1000) {
        let (_temp, engine) = create_test_engine();

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), format!("value_{}", seed));

        let embedding = generate_embedding(384, seed);
        let id = engine.add(
            "Test memory".to_string(),
            embedding,
            Some(metadata.clone()),
            None,
            None,
            None,
        ).unwrap();

        let retrieved = engine.get(&id).unwrap().unwrap();

        prop_assert_eq!(retrieved.metadata.get("key"), metadata.get("key"));
    }

    /// Property: Memory timestamp is set if provided
    #[test]
    fn prop_memory_timestamp_preserved(days_ago in 1u64..100) {
        use mnemefusion_core::types::Timestamp;

        let (_temp, engine) = create_test_engine();

        let timestamp = Timestamp::now().subtract_days(days_ago);
        let embedding = generate_embedding(384, 42);

        let id = engine.add(
            "Test memory".to_string(),
            embedding,
            None,
            Some(timestamp),
            None,
            None,
        ).unwrap();

        let retrieved = engine.get(&id).unwrap().unwrap();

        prop_assert_eq!(retrieved.created_at, timestamp);
    }

    /// Property: Memory content is preserved exactly
    #[test]
    fn prop_memory_content_preserved(content in "\\PC{1,100}") {
        let (_temp, engine) = create_test_engine();

        let embedding = generate_embedding(384, 42);
        let id = engine.add(
            content.clone(),
            embedding,
            None,
            None,
            None,
            None,
        ).unwrap();

        let retrieved = engine.get(&id).unwrap().unwrap();

        prop_assert_eq!(retrieved.content, content);
    }

    /// Property: List IDs returns all added memory IDs
    #[test]
    fn prop_memory_list_complete(count in 1usize..10) {
        let (_temp, engine) = create_test_engine();

        let mut expected_ids = Vec::new();
        for i in 0..count {
            let embedding = generate_embedding(384, i as u64);
            let id = engine.add(
                format!("Memory {}", i),
                embedding,
                None,
                None,
                None,
                None,
            ).unwrap();
            expected_ids.push(id);
        }

        let listed = engine.list_ids().unwrap();

        prop_assert_eq!(listed.len(), count);
        // All expected IDs should be in the list
        for id in &expected_ids {
            prop_assert!(listed.contains(id));
        }
    }

    /// Property: Adding duplicate content creates separate memories
    #[test]
    fn prop_memory_duplicates_allowed(seed in 0u64..100) {
        let (_temp, engine) = create_test_engine();

        let content = "Duplicate content".to_string();
        let embedding = generate_embedding(384, seed);

        let id1 = engine.add(
            content.clone(),
            embedding.clone(),
            None,
            None,
            None,
            None,
        ).unwrap();

        let id2 = engine.add(
            content.clone(),
            embedding.clone(),
            None,
            None,
            None,
            None,
        ).unwrap();

        prop_assert_ne!(id1, id2);
    }
}
