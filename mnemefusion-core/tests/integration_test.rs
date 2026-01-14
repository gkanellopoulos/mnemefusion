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

        let id = engine.add(content.clone(), embedding.clone(), None, None).unwrap();

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
            .add(content.to_string(), embedding, Some(metadata), None)
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
        let id = engine.add(content, embedding, None, None).unwrap();
        ids.push(id);
    }

    assert_eq!(engine.count().unwrap(), 5);

    // Delete the middle one
    let deleted = engine.delete(&ids[2]).unwrap();
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
    let deleted = engine.delete(&ids[2]).unwrap();
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
    let result = engine.add("test".to_string(), vec![0.1; 384], None, None);
    assert!(result.is_err());

    // Add with correct dimension
    let id = engine
        .add("test".to_string(), vec![0.1; 512], None, None)
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
        engine.add(content, embedding, None, None).unwrap();
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
        .add("2021 memory".to_string(), vec![0.1; 384], None, Some(ts1))
        .unwrap();
    let id2 = engine
        .add("2022 memory".to_string(), vec![0.2; 384], None, Some(ts2))
        .unwrap();
    let id3 = engine
        .add("2023 memory".to_string(), vec![0.3; 384], None, Some(ts3))
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
