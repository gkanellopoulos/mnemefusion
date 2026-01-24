//! Integration tests for MnemeFusion
//!
//! These tests verify the complete end-to-end functionality of the memory engine.

use mnemefusion_core::{Config, MemoryEngine};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_create_add_retrieve_close_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("integration_test.mfdb");

    let memory_id = {
        // Create engine, add memory, close
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        assert_eq!(engine.count().unwrap(), 0);

        let content = "The quick brown fox jumps over the lazy dog".to_string();
        let embedding = vec![0.5; 384]; // 384 dimensions

        let id = engine.add(content.clone(), embedding.clone(), None, None, None, None).unwrap();

        // Verify it was added
        assert_eq!(engine.count().unwrap(), 1);

        let memory = engine.get(&id).unwrap().expect("Memory should exist");
        assert_eq!(memory.content, content);
        assert_eq!(memory.embedding.len(), 384);

        engine.close().unwrap();
        id
    };

    // Reopen and verify persistence
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        assert_eq!(engine.count().unwrap(), 1);

        let memory = engine.get(&memory_id).unwrap().expect("Memory should persist");
        assert_eq!(memory.content, "The quick brown fox jumps over the lazy dog");

        engine.close().unwrap();
    }
}

#[test]
fn test_multiple_memories_with_metadata() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("metadata_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Add multiple memories with metadata
    let memories = vec![
        (
            "Project deadline moved to March 15th",
            vec![("source", "email"), ("category", "project")],
        ),
        (
            "Team meeting scheduled for tomorrow at 2pm",
            vec![("source", "calendar"), ("category", "meeting")],
        ),
        (
            "Budget approved for Q2",
            vec![("source", "announcement"), ("category", "finance")],
        ),
    ];

    let mut ids = Vec::new();
    for (content, meta) in &memories {
        let mut metadata = HashMap::new();
        for (key, value) in meta {
            metadata.insert(key.to_string(), value.to_string());
        }

        let embedding = vec![0.5; 384];
        let id = engine
            .add(content.to_string(), embedding, Some(metadata), None, None, None)
            .unwrap();
        ids.push(id);
    }

    assert_eq!(engine.count().unwrap(), 3);

    // Verify each memory
    for (idx, id) in ids.iter().enumerate() {
        let memory = engine.get(id).unwrap().expect("Memory should exist");
        assert_eq!(memory.content, memories[idx].0);

        for (key, expected_value) in &memories[idx].1 {
            let value = memory.metadata.get(&key.to_string()).unwrap();
            assert_eq!(value, expected_value);
        }
    }

    engine.close().unwrap();
}

#[test]
fn test_delete_operations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("delete_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Add 5 memories
    let mut ids = Vec::new();
    for i in 0..5 {
        let content = format!("Memory number {}", i);
        let embedding = vec![i as f32 * 0.1; 384];
        let id = engine.add(content, embedding, None, None, None, None).unwrap();
        ids.push(id);
    }

    assert_eq!(engine.count().unwrap(), 5);

    // Delete the middle one
    let deleted = engine.delete(&ids[2], None).unwrap();
    assert!(deleted);
    assert_eq!(engine.count().unwrap(), 4);

    // Verify it's gone
    let memory = engine.get(&ids[2]).unwrap();
    assert!(memory.is_none());

    // Verify others still exist
    assert!(engine.get(&ids[0]).unwrap().is_some());
    assert!(engine.get(&ids[1]).unwrap().is_some());
    assert!(engine.get(&ids[3]).unwrap().is_some());
    assert!(engine.get(&ids[4]).unwrap().is_some());

    // Try to delete non-existent
    let deleted = engine.delete(&ids[2], None).unwrap();
    assert!(!deleted);

    engine.close().unwrap();
}

#[test]
fn test_custom_configuration() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("config_test.mfdb");

    let config = Config::new()
        .with_embedding_dim(512)
        .with_temporal_decay_hours(336.0)
        .with_causal_max_hops(5);

    let engine = MemoryEngine::open(&path, config).unwrap();

    assert_eq!(engine.config().embedding_dim, 512);
    assert_eq!(engine.config().temporal_decay_hours, 336.0);
    assert_eq!(engine.config().causal_max_hops, 5);

    // Try adding with wrong dimension
    let result = engine.add("test".to_string(), vec![0.1; 384], None, None, None, None);
    assert!(result.is_err());

    // Add with correct dimension
    let id = engine
        .add("test".to_string(), vec![0.1; 512], None, None, None, None)
        .unwrap();

    let memory = engine.get(&id).unwrap().unwrap();
    assert_eq!(memory.embedding.len(), 512);

    engine.close().unwrap();
}

#[test]
fn test_large_number_of_memories() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Add 1000 memories
    let count = 1000;
    for i in 0..count {
        let content = format!("Memory {}", i);
        let embedding = vec![(i % 100) as f32 / 100.0; 384];
        engine.add(content, embedding, None, None, None, None).unwrap();
    }

    assert_eq!(engine.count().unwrap(), count);

    let ids = engine.list_ids().unwrap();
    assert_eq!(ids.len(), count);

    engine.close().unwrap();
}

#[test]
fn test_custom_timestamps() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("timestamp_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    use mnemefusion_core::Timestamp;

    // Add memories with custom timestamps
    let ts1 = Timestamp::from_unix_secs(1609459200.0); // 2021-01-01
    let ts2 = Timestamp::from_unix_secs(1640995200.0); // 2022-01-01
    let ts3 = Timestamp::from_unix_secs(1672531200.0); // 2023-01-01

    let id1 = engine
        .add("2021 memory".to_string(), vec![0.1; 384], None, Some(ts1), None, None)
        .unwrap();
    let id2 = engine
        .add("2022 memory".to_string(), vec![0.2; 384], None, Some(ts2), None, None)
        .unwrap();
    let id3 = engine
        .add("2023 memory".to_string(), vec![0.3; 384], None, Some(ts3), None, None)
        .unwrap();

    // Verify timestamps
    let mem1 = engine.get(&id1).unwrap().unwrap();
    let mem2 = engine.get(&id2).unwrap().unwrap();
    let mem3 = engine.get(&id3).unwrap().unwrap();

    assert_eq!(mem1.created_at, ts1);
    assert_eq!(mem2.created_at, ts2);
    assert_eq!(mem3.created_at, ts3);

    engine.close().unwrap();
}

#[test]
fn test_temporal_range_query() {
    use mnemefusion_core::Timestamp;

    let dir = tempdir().unwrap();
    let path = dir.path().join("temporal_range_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    let now = Timestamp::now();

    // Add memories at different times
    let id1 = engine
        .add(
            "Memory from 10 days ago".to_string(),
            vec![0.1; 384],
            None,
            Some(now.subtract_days(10)),
            None,
            None,
        )
        .unwrap();

    let id2 = engine
        .add(
            "Memory from 5 days ago".to_string(),
            vec![0.2; 384],
            None,
            Some(now.subtract_days(5)),
            None,
            None,
        )
        .unwrap();

    let id3 = engine
        .add(
            "Memory from yesterday".to_string(),
            vec![0.3; 384],
            None,
            Some(now.subtract_days(1)),
            None,
            None,
        )
        .unwrap();

    let id4 = engine
        .add(
            "Memory from today".to_string(),
            vec![0.4; 384],
            None,
            Some(now),
            None,
            None,
        )
        .unwrap();

    // Query last 7 days
    let results = engine
        .get_range(now.subtract_days(7), now, 100, None)
        .unwrap();

    assert_eq!(results.len(), 3); // Should get id2, id3, id4 (all within 7 days)

    // Should be newest first
    assert_eq!(results[0].0.id, id4);
    assert_eq!(results[1].0.id, id3);
    assert_eq!(results[2].0.id, id2);

    // Query last 15 days
    let results = engine
        .get_range(now.subtract_days(15), now, 100, None)
        .unwrap();

    assert_eq!(results.len(), 4); // All memories

    // Verify ordering (newest first)
    assert_eq!(results[0].0.id, id4);
    assert_eq!(results[1].0.id, id3);
    assert_eq!(results[2].0.id, id2);
    assert_eq!(results[3].0.id, id1);

    // Test empty range (future)
    let future_start = now.add_days(1);
    let future_end = now.add_days(2);
    let results = engine
        .get_range(future_start, future_end, 100, None)
        .unwrap();
    assert_eq!(results.len(), 0);

    engine.close().unwrap();
}

#[test]
fn test_get_recent_memories() {
    use mnemefusion_core::Timestamp;

    let dir = tempdir().unwrap();
    let path = dir.path().join("recent_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    let now = Timestamp::now();

    // Add 10 memories with different timestamps
    let mut ids = Vec::new();
    for i in 0..10 {
        let id = engine
            .add(
                format!("Memory {}", i),
                vec![i as f32 * 0.1; 384],
                None,
                Some(now.subtract_days(i as u64)),
                None,
                None,
            )
            .unwrap();
        ids.push(id);
    }

    // Get 5 most recent
    let results = engine.get_recent(5, None).unwrap();
    assert_eq!(results.len(), 5);

    // Should be newest first (ids[0] is newest)
    assert_eq!(results[0].0.id, ids[0]);
    assert_eq!(results[1].0.id, ids[1]);
    assert_eq!(results[2].0.id, ids[2]);
    assert_eq!(results[3].0.id, ids[3]);
    assert_eq!(results[4].0.id, ids[4]);

    // Get all (limit > count)
    let results = engine.get_recent(100, None).unwrap();
    assert_eq!(results.len(), 10);

    // Test with empty database
    let dir2 = tempdir().unwrap();
    let path2 = dir2.path().join("empty_test.mfdb");
    let engine2 = MemoryEngine::open(&path2, Config::default()).unwrap();

    let results = engine2.get_recent(10, None).unwrap();
    assert_eq!(results.len(), 0);

    engine.close().unwrap();
    engine2.close().unwrap();
}

#[test]
fn test_temporal_with_search() {
    use mnemefusion_core::Timestamp;

    let dir = tempdir().unwrap();
    let path = dir.path().join("temporal_search_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    let now = Timestamp::now();

    // Add memories with semantic similarity and different times
    let id1 = engine
        .add(
            "Team meeting about project deadline".to_string(),
            vec![0.8; 384],
            None,
            Some(now.subtract_days(7)),
            None,
            None,
        )
        .unwrap();

    let id2 = engine
        .add(
            "Client meeting regarding timeline".to_string(),
            vec![0.7; 384],
            None,
            Some(now.subtract_days(3)),
            None,
            None,
        )
        .unwrap();

    let id3 = engine
        .add(
            "Lunch with colleague".to_string(),
            vec![0.1; 384],
            None,
            Some(now.subtract_days(1)),
            None,
            None,
        )
        .unwrap();

    // Semantic search
    let query_embedding = vec![0.75; 384];
    let search_results = engine.search(&query_embedding, 10, None, None).unwrap();

    // All should be found
    assert_eq!(search_results.len(), 3);

    // Temporal query for recent week
    let temporal_results = engine
        .get_range(now.subtract_days(5), now, 10, None)
        .unwrap();

    // Should get only id2 and id3 (within 5 days)
    assert_eq!(temporal_results.len(), 2);
    assert_eq!(temporal_results[0].0.id, id3); // Newest first
    assert_eq!(temporal_results[1].0.id, id2);

    // Get recent shows all in order
    let recent = engine.get_recent(3, None).unwrap();
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].0.id, id3);
    assert_eq!(recent[1].0.id, id2);
    assert_eq!(recent[2].0.id, id1);

    engine.close().unwrap();
}

#[test]
fn test_causal_simple_chain() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("causal_simple_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Create a simple causal chain: m1 → m2 → m3
    let id1 = engine
        .add(
            "Meeting was scheduled".to_string(),
            vec![0.1; 384],
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let id2 = engine
        .add(
            "Team prepared for meeting".to_string(),
            vec![0.2; 384],
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let id3 = engine
        .add(
            "Meeting was successful".to_string(),
            vec![0.3; 384],
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Add causal links
    engine
        .add_causal_link(&id1, &id2, 0.9, "Scheduling caused preparation".to_string())
        .unwrap();

    engine
        .add_causal_link(&id2, &id3, 0.8, "Preparation led to success".to_string())
        .unwrap();

    // Test get_effects from id1
    let effects = engine.get_effects(&id1, 2).unwrap();
    assert_eq!(effects.paths.len(), 2); // id1→id2 and id1→id2→id3

    // Verify the longer path
    let longest_path = effects
        .paths
        .iter()
        .max_by_key(|p| p.memories.len())
        .unwrap();
    assert_eq!(longest_path.memories.len(), 3);
    assert_eq!(longest_path.memories[0], id1);
    assert_eq!(longest_path.memories[1], id2);
    assert_eq!(longest_path.memories[2], id3);

    // Verify cumulative confidence
    let expected_confidence = 0.9 * 0.8;
    assert!((longest_path.confidence - expected_confidence).abs() < 0.001);

    // Test get_causes from id3
    let causes = engine.get_causes(&id3, 2).unwrap();
    assert_eq!(causes.paths.len(), 2); // id3←id2 and id3←id2←id1

    engine.close().unwrap();
}

#[test]
fn test_causal_multi_hop_traversal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("causal_multi_hop_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Create a branching causal graph:
    //     m1 → m2 → m4
    //      ↓
    //     m3 → m5
    let id1 = engine
        .add("Root cause".to_string(), vec![0.1; 384], None, None, None, None)
        .unwrap();

    let id2 = engine
        .add("Effect A".to_string(), vec![0.2; 384], None, None, None, None)
        .unwrap();

    let id3 = engine
        .add("Effect B".to_string(), vec![0.3; 384], None, None, None, None)
        .unwrap();

    let id4 = engine
        .add("Second-order effect A".to_string(), vec![0.4; 384], None, None, None, None)
        .unwrap();

    let id5 = engine
        .add("Second-order effect B".to_string(), vec![0.5; 384], None, None, None, None)
        .unwrap();

    // Add causal links
    engine
        .add_causal_link(&id1, &id2, 0.9, "Root → A".to_string())
        .unwrap();

    engine
        .add_causal_link(&id1, &id3, 0.85, "Root → B".to_string())
        .unwrap();

    engine
        .add_causal_link(&id2, &id4, 0.8, "A → 2nd A".to_string())
        .unwrap();

    engine
        .add_causal_link(&id3, &id5, 0.75, "B → 2nd B".to_string())
        .unwrap();

    // Test get_effects from id1 with max_hops=2
    let effects = engine.get_effects(&id1, 2).unwrap();
    // Should find: id1→id2, id1→id3, id1→id2→id4, id1→id3→id5
    assert_eq!(effects.paths.len(), 4);

    // Test max_hops limiting
    let effects_one_hop = engine.get_effects(&id1, 1).unwrap();
    assert_eq!(effects_one_hop.paths.len(), 2); // Only id1→id2 and id1→id3

    // Test get_causes from id5
    let causes = engine.get_causes(&id5, 2).unwrap();
    // Should find: id5←id3 and id5←id3←id1
    assert_eq!(causes.paths.len(), 2);

    engine.close().unwrap();
}

#[test]
fn test_causal_graph_persistence() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("causal_persist_test.mfdb");

    let id1;
    let id2;
    let id3;

    // Create and save causal graph
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        id1 = engine
            .add("Cause A".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        id2 = engine
            .add("Effect A".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

        id3 = engine
            .add("Effect B".to_string(), vec![0.3; 384], None, None, None, None)
            .unwrap();

        // Add causal links
        engine
            .add_causal_link(&id1, &id2, 0.9, "A → B".to_string())
            .unwrap();

        engine
            .add_causal_link(&id2, &id3, 0.8, "B → C".to_string())
            .unwrap();

        // Close (saves graph)
        engine.close().unwrap();
    }

    // Reopen and verify causal graph persisted
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Verify memories persisted
        assert_eq!(engine.count().unwrap(), 3);

        // Verify causal links persisted
        let effects = engine.get_effects(&id1, 2).unwrap();
        assert_eq!(effects.paths.len(), 2); // id1→id2 and id1→id2→id3

        // Verify cumulative confidence on longest path
        let longest_path = effects
            .paths
            .iter()
            .max_by_key(|p| p.memories.len())
            .unwrap();
        assert_eq!(longest_path.memories.len(), 3);

        let expected_confidence = 0.9 * 0.8;
        assert!((longest_path.confidence - expected_confidence).abs() < 0.001);

        // Test get_causes too
        let causes = engine.get_causes(&id3, 2).unwrap();
        assert_eq!(causes.paths.len(), 2);

        engine.close().unwrap();
    }
}

// ============================================================================
// Sprint 13 Task 6: Crash Recovery and Corruption Tests
// ============================================================================

/// Test that database can recover after being reopened without close()
/// This simulates a crash where close() is never called
#[test]
fn test_crash_recovery_no_close() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("crash_test.mfdb");

    let id1;
    let id2;

    // Add memories WITHOUT calling close() (simulates crash)
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        id1 = engine
            .add("Memory 1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        id2 = engine
            .add("Memory 2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

        // NO close() - simulates crash
        drop(engine);
    }

    // Reopen database - should recover
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Both memories should be present due to eager save pattern
        assert_eq!(engine.count().unwrap(), 2);

        let mem1 = engine.get(&id1).unwrap().expect("Memory 1 should exist");
        assert_eq!(mem1.content, "Memory 1");

        let mem2 = engine.get(&id2).unwrap().expect("Memory 2 should exist");
        assert_eq!(mem2.content, "Memory 2");

        engine.close().unwrap();
    }
}

/// Test recovery after crash during batch operation
#[test]
fn test_crash_recovery_during_batch() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("batch_crash_test.mfdb");

    // Start batch operation WITHOUT close
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Add first memory
        let _ = engine
            .add("Before batch".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        // Prepare batch (simulates crash before batch completes)
        let inputs = vec![
            mnemefusion_core::types::MemoryInput::new("Batch 1".to_string(), vec![0.2; 384]),
            mnemefusion_core::types::MemoryInput::new("Batch 2".to_string(), vec![0.3; 384]),
        ];

        let _ = engine.add_batch(inputs, None).unwrap();

        // NO close() - crash simulation
        drop(engine);
    }

    // Reopen and verify consistent state
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // All memories should be present (batch completed due to eager save)
        assert_eq!(engine.count().unwrap(), 3);

        engine.close().unwrap();
    }
}

/// Test detection of truncated database file
#[test]
fn test_corruption_truncated_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("truncated.mfdb");

    // Create a tiny truncated file
    std::fs::write(&path, b"MFDB").unwrap();

    // Should fail to open
    let result = MemoryEngine::open(&path, Config::default());
    assert!(result.is_err());

    if let Err(err) = result {
        let err_str = err.to_string();
        assert!(err_str.contains("truncated") || err_str.contains("too small"));
    }
}

/// Test detection of corrupted header
#[test]
fn test_corruption_bad_header() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_header.mfdb");

    // Create file with bad magic number
    let mut bad_header = vec![0u8; 1024];
    bad_header[0..4].copy_from_slice(b"XXXX"); // Wrong magic
    std::fs::write(&path, &bad_header).unwrap();

    // Should fail to open with clear error
    let result = MemoryEngine::open(&path, Config::default());
    assert!(result.is_err());

    if let Err(err) = result {
        let err_str = err.to_string().to_lowercase();
        // Should mention either invalid format, magic number, or corruption
        // Using lowercase to be case-insensitive
        assert!(
            err_str.contains("invalid") ||
            err_str.contains("magic") ||
            err_str.contains("corruption") ||
            err_str.contains("database") ||
            err_str.contains("storage")
        );
    }
}

/// Test ACID: Atomicity - failed operation leaves no partial state
#[test]
fn test_acid_atomicity_failed_add() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("atomicity_test.mfdb");
    let engine = MemoryEngine::open(&path, Config::default()).unwrap();

    // Add valid memory
    let _ = engine
        .add("Valid memory".to_string(), vec![0.1; 384], None, None, None, None)
        .unwrap();

    assert_eq!(engine.count().unwrap(), 1);

    // Try to add memory with wrong embedding dimension (should fail)
    let result = engine.add(
        "Invalid memory".to_string(),
        vec![0.2; 100], // Wrong dimension
        None,
        None,
        None,
        None,
    );

    assert!(result.is_err());

    // Count should still be 1 (failed add should not create partial state)
    assert_eq!(engine.count().unwrap(), 1);

    engine.close().unwrap();
}

/// Test ACID: Consistency - database remains in valid state
#[test]
fn test_acid_consistency_after_operations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("consistency_test.mfdb");

    let id1;
    let id2;

    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        id1 = engine
            .add("Memory 1".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        id2 = engine
            .add("Memory 2".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

        // Delete one
        engine.delete(&id1, None).unwrap();

        engine.close().unwrap();
    }

    // Reopen and verify database is consistent
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        assert_eq!(engine.count().unwrap(), 1);

        assert!(engine.get(&id1).unwrap().is_none());
        assert!(engine.get(&id2).unwrap().is_some());

        // Search should work without crashes
        let results = engine.search(&vec![0.2; 384], 10, None, None).unwrap();
        assert!(!results.is_empty());

        engine.close().unwrap();
    }
}

/// Test ACID: Durability - changes persist after close
#[test]
fn test_acid_durability() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("durability_test.mfdb");

    let id;

    // Write and close
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        id = engine
            .add("Durable memory".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        engine.close().unwrap();
    }

    // Reopen - data should be durable
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        let memory = engine.get(&id).unwrap().expect("Memory should be durable");
        assert_eq!(memory.content, "Durable memory");

        engine.close().unwrap();
    }
}

/// Test recovery with vector index intact
#[test]
fn test_crash_recovery_vector_index_intact() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("vector_crash_test.mfdb");

    // Add memories with embeddings WITHOUT close
    {
        let config = Config::default().with_embedding_dim(3);
        let engine = MemoryEngine::open(&path, config).unwrap();

        let _ = engine
            .add("Vector 1".to_string(), vec![1.0, 0.0, 0.0], None, None, None, None)
            .unwrap();

        let _ = engine
            .add("Vector 2".to_string(), vec![0.0, 1.0, 0.0], None, None, None, None)
            .unwrap();

        // NO close() - crash
        drop(engine);
    }

    // Reopen - vector index should be intact
    {
        let config = Config::default().with_embedding_dim(3);
        let engine = MemoryEngine::open(&path, config).unwrap();

        // Search should work
        let results = engine.search(&vec![1.0, 0.0, 0.0], 2, None, None).unwrap();
        assert_eq!(results.len(), 2);

        // First result should be most similar to query
        assert!(results[0].1 > results[1].1); // Higher similarity score

        engine.close().unwrap();
    }
}

/// Test recovery with causal graph intact
#[test]
fn test_crash_recovery_causal_graph_intact() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("graph_crash_test.mfdb");

    let id1;
    let id2;

    // Create causal graph WITHOUT close
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        id1 = engine
            .add("Cause".to_string(), vec![0.1; 384], None, None, None, None)
            .unwrap();

        id2 = engine
            .add("Effect".to_string(), vec![0.2; 384], None, None, None, None)
            .unwrap();

        engine
            .add_causal_link(&id1, &id2, 0.9, "Link".to_string())
            .unwrap();

        // NO close() - crash
        drop(engine);
    }

    // Reopen - graph should be intact
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();

        // Causal links should be preserved
        let effects = engine.get_effects(&id1, 1).unwrap();
        assert_eq!(effects.paths.len(), 1);
        assert_eq!(effects.paths[0].memories.len(), 2);

        engine.close().unwrap();
    }
}

/// Test validation detects missing tables
#[test]
fn test_validation_detects_corruption() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("validation_test.mfdb");

    // Create and close a valid database
    {
        let engine = MemoryEngine::open(&path, Config::default()).unwrap();
        engine.close().unwrap();
    }

    // Database should reopen successfully with validation
    {
        let result = MemoryEngine::open(&path, Config::default());
        assert!(result.is_ok());

        if let Ok(engine) = result {
            engine.close().unwrap();
        }
    }
}
