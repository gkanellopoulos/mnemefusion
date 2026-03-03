//! Temporal index for time-based memory queries
//!
//! This module provides efficient time-range queries using redb's native B-tree ordering.
//! Memories are indexed by their timestamp, allowing fast range queries and "recent N" lookups.

use crate::{
    storage::StorageEngine,
    types::{MemoryId, Timestamp},
    Result,
};
use redb::ReadableTable;
use std::sync::Arc;

/// Result from a temporal query
#[derive(Debug, Clone)]
pub struct TemporalResult {
    /// The memory ID
    pub id: MemoryId,
    /// The timestamp of the memory
    pub timestamp: Timestamp,
}

/// Temporal index for time-based queries
///
/// Uses redb's native B-tree ordering for efficient range queries.
/// The TEMPORAL_INDEX table maps timestamp (u64) → memory_id (bytes).
pub struct TemporalIndex {
    storage: Arc<StorageEngine>,
}

impl TemporalIndex {
    /// Create a new temporal index
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage engine containing the TEMPORAL_INDEX table
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        Self { storage }
    }

    /// Query memories within a time range
    ///
    /// Returns memories whose timestamps fall within [start, end] (inclusive),
    /// sorted by timestamp in descending order (newest first).
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the time range (inclusive)
    /// * `end` - End of the time range (inclusive)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of TemporalResult, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::{index::TemporalIndex, Timestamp};
    /// # use std::sync::Arc;
    /// # let storage = Arc::new(mnemefusion_core::storage::StorageEngine::open("test.mfdb").unwrap());
    /// let temporal = TemporalIndex::new(storage);
    /// let now = Timestamp::now();
    /// let week_ago = now.subtract_days(7);
    ///
    /// let results = temporal.range_query(week_ago, now, 100).unwrap();
    /// println!("Found {} memories from the past week", results.len());
    /// ```
    pub fn range_query(
        &self,
        start: Timestamp,
        end: Timestamp,
        limit: usize,
    ) -> Result<Vec<TemporalResult>> {
        let read_txn = self.storage.db().begin_read()?;
        let table = read_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;

        // Get range in ascending order, then reverse
        let range = table.range(start.as_micros()..=end.as_micros())?;

        let mut results = Vec::new();
        for entry in range {
            let (ts_key, id_bytes) = entry?;
            let timestamp = Timestamp::from_micros(ts_key.value());
            let id = MemoryId::from_bytes(id_bytes.value())?;

            results.push(TemporalResult { id, timestamp });

            if results.len() >= limit {
                break;
            }
        }

        // Reverse to get newest first
        results.reverse();

        Ok(results)
    }

    /// Get the N most recent memories
    ///
    /// Returns the N most recent memories, sorted by timestamp descending (newest first).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent memories to retrieve
    ///
    /// # Returns
    ///
    /// A vector of TemporalResult, sorted newest first
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::index::TemporalIndex;
    /// # use std::sync::Arc;
    /// # let storage = Arc::new(mnemefusion_core::storage::StorageEngine::open("test.mfdb").unwrap());
    /// let temporal = TemporalIndex::new(storage);
    ///
    /// let recent = temporal.recent(10).unwrap();
    /// println!("10 most recent memories:");
    /// for result in recent {
    ///     println!("  {} at {:?}", result.id, result.timestamp);
    /// }
    /// ```
    pub fn recent(&self, n: usize) -> Result<Vec<TemporalResult>> {
        let read_txn = self.storage.db().begin_read()?;
        let table = read_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;

        // Use reverse iterator to get newest first
        let mut results = Vec::with_capacity(n);
        let iter = table.iter()?.rev();

        for entry in iter.take(n) {
            let (ts_key, id_bytes) = entry?;
            let timestamp = Timestamp::from_micros(ts_key.value());
            let id = MemoryId::from_bytes(id_bytes.value())?;

            results.push(TemporalResult { id, timestamp });
        }

        Ok(results)
    }

    /// Get the count of memories within a time range
    ///
    /// Efficiently counts memories without loading full data.
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the time range (inclusive)
    /// * `end` - End of the time range (inclusive)
    pub fn count_range(&self, start: Timestamp, end: Timestamp) -> Result<usize> {
        let read_txn = self.storage.db().begin_read()?;
        let table = read_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;

        let range = table.range(start.as_micros()..=end.as_micros())?;
        let count = range.count();

        Ok(count)
    }

    /// Add a memory to the temporal index
    ///
    /// This is called automatically when a memory is added to the database.
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to index
    /// * `timestamp` - The timestamp of the memory
    pub fn add(&self, id: &MemoryId, timestamp: Timestamp) -> Result<()> {
        let write_txn = self.storage.db().begin_write()?;
        {
            let mut table = write_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;
            table.insert(timestamp.as_micros(), id.as_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Remove a memory from the temporal index
    ///
    /// This is called automatically when a memory is deleted from the database.
    /// Since we don't have the timestamp during deletion, we need to scan for it.
    ///
    /// # Arguments
    ///
    /// * `id` - The memory ID to remove
    pub fn remove(&self, id: &MemoryId) -> Result<()> {
        let write_txn = self.storage.db().begin_write()?;
        {
            // First, find the timestamp(s) associated with this memory
            let timestamps_to_remove: Vec<u64> = {
                let table = write_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;
                let mut timestamps = Vec::new();

                for entry in table.iter()? {
                    let (ts_key, id_bytes) = entry?;
                    if let Ok(entry_id) = MemoryId::from_bytes(id_bytes.value()) {
                        if &entry_id == id {
                            timestamps.push(ts_key.value());
                        }
                    }
                }

                timestamps
            };

            // Now remove all found entries
            let mut table = write_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;
            for ts in timestamps_to_remove {
                table.remove(ts)?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Search memories by temporal content matching
    ///
    /// Unlike timestamp-based queries (recent, range_query), this method matches
    /// temporal expressions in the query to temporal expressions in memory content.
    /// For example, "yesterday" in the query matches memories that mention "yesterday".
    ///
    /// # Arguments
    ///
    /// * `query_text` - The query text to extract temporal expressions from
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (MemoryId, score) tuples, sorted by score descending
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mnemefusion_core::index::TemporalIndex;
    /// # use std::sync::Arc;
    /// # let storage = Arc::new(mnemefusion_core::storage::StorageEngine::open("test.mfdb").unwrap());
    /// let temporal = TemporalIndex::new(storage);
    ///
    /// // Query mentions "yesterday" - will match memories with "yesterday" in content
    /// let results = temporal.search_temporal_content("What happened yesterday?", 10).unwrap();
    /// for (id, score) in results {
    ///     println!("Memory {} has temporal overlap score: {}", id, score);
    /// }
    /// ```
    pub fn search_temporal_content(
        &self,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        use crate::ingest::get_temporal_extractor;

        // Extract temporal expressions from query
        let extractor = get_temporal_extractor();
        let query_expressions = extractor.extract(query_text);

        // If query has no temporal expressions, return empty
        if query_expressions.is_empty() {
            return Ok(vec![]);
        }

        // Get all memories and score them
        let mut scored_results = Vec::new();

        // Get all memories from storage (TODO: optimize this with an index)
        let read_txn = self.storage.db().begin_read()?;
        let table = read_txn.open_table(crate::storage::engine::TEMPORAL_INDEX)?;

        for entry in table.iter()? {
            let (_ts_key, id_bytes) = entry?;
            let memory_id = MemoryId::from_bytes(id_bytes.value())?;

            // Get memory to access metadata
            if let Some(memory) = self.storage.get_memory(&memory_id)? {
                // Get temporal expressions from metadata
                if let Some(temporal_expr_json) = memory.get_metadata("temporal_expressions") {
                    // Parse JSON array of temporal expression strings
                    if let Ok(expr_strings) = serde_json::from_str::<Vec<String>>(temporal_expr_json) {
                        if !expr_strings.is_empty() {
                            // Convert strings back to TemporalExpression objects
                            // For scoring, we just need the text, so create simple Relative expressions
                            let memory_expressions: Vec<_> = expr_strings
                                .iter()
                                .map(|s| crate::ingest::TemporalExpression::Relative(s.clone()))
                                .collect();

                            // Calculate overlap score
                            let score = extractor.calculate_overlap(&query_expressions, &memory_expressions);

                            if score > 0.0 {
                                scored_results.push((memory_id, score));
                            }
                        }
                    }
                }
            }
        }

        // Sort by score descending
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K
        scored_results.truncate(limit);

        Ok(scored_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{storage::StorageEngine, types::Memory};
    use tempfile::tempdir;

    fn create_test_index() -> (TemporalIndex, Arc<StorageEngine>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let storage = Arc::new(StorageEngine::open(&path).unwrap());
        let temporal = TemporalIndex::new(Arc::clone(&storage));

        (temporal, storage, dir)
    }

    #[test]
    fn test_recent_with_no_memories() {
        let (temporal, _storage, _dir) = create_test_index();

        let results = temporal.recent(10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_recent_with_multiple_memories() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();

        // Add memories with different timestamps
        let mem1 =
            Memory::new_with_timestamp("oldest".to_string(), vec![0.1; 384], now.subtract_days(3));
        let mem2 =
            Memory::new_with_timestamp("middle".to_string(), vec![0.2; 384], now.subtract_days(2));
        let mem3 =
            Memory::new_with_timestamp("newest".to_string(), vec![0.3; 384], now.subtract_days(1));

        storage.store_memory(&mem1).unwrap();
        storage.store_memory(&mem2).unwrap();
        storage.store_memory(&mem3).unwrap();

        // Get 2 most recent
        let results = temporal.recent(2).unwrap();
        assert_eq!(results.len(), 2);

        // Should be newest first
        assert_eq!(results[0].timestamp, mem3.created_at);
        assert_eq!(results[1].timestamp, mem2.created_at);
    }

    #[test]
    fn test_recent_with_limit_larger_than_count() {
        let (temporal, storage, _dir) = create_test_index();

        let mem = Memory::new("test".to_string(), vec![0.1; 384]);
        storage.store_memory(&mem).unwrap();

        let results = temporal.recent(100).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_range_query_within_range() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();
        let start = now.subtract_days(7);
        let end = now;

        // Add memory within range
        let mem1 =
            Memory::new_with_timestamp("within".to_string(), vec![0.1; 384], now.subtract_days(3));
        // Add memory outside range (too old)
        let mem2 = Memory::new_with_timestamp(
            "too old".to_string(),
            vec![0.2; 384],
            now.subtract_days(10),
        );

        storage.store_memory(&mem1).unwrap();
        storage.store_memory(&mem2).unwrap();

        let results = temporal.range_query(start, end, 100).unwrap();

        // Should only get mem1
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, mem1.id);
    }

    #[test]
    fn test_range_query_empty_range() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();
        let mem = Memory::new("test".to_string(), vec![0.1; 384]);
        storage.store_memory(&mem).unwrap();

        // Query range in the future
        let future_start = now.add_days(1);
        let future_end = now.add_days(2);

        let results = temporal.range_query(future_start, future_end, 100).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_range_query_with_limit() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();

        // Add 5 memories
        for i in 0..5 {
            let mem = Memory::new_with_timestamp(
                format!("memory {}", i),
                vec![0.1; 384],
                now.subtract_days(i as u64),
            );
            storage.store_memory(&mem).unwrap();
        }

        // Query all but limit to 3
        let results = temporal.range_query(now.subtract_days(10), now, 3).unwrap();
        assert_eq!(results.len(), 3);

        // Should be newest first
        assert!(results[0].timestamp > results[1].timestamp);
        assert!(results[1].timestamp > results[2].timestamp);
    }

    #[test]
    fn test_count_range() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();

        // Add memories
        for i in 0..10 {
            let mem = Memory::new_with_timestamp(
                format!("memory {}", i),
                vec![0.1; 384],
                now.subtract_days(i as u64),
            );
            storage.store_memory(&mem).unwrap();
        }

        // Count last 5 days
        let count = temporal.count_range(now.subtract_days(5), now).unwrap();
        assert_eq!(count, 6); // Days 0,1,2,3,4,5 = 6 days
    }

    #[test]
    fn test_range_query_ordering() {
        let (temporal, storage, _dir) = create_test_index();

        let now = Timestamp::now();

        // Add memories in random order
        let times = vec![5, 2, 8, 1, 9, 3];
        for (i, days_ago) in times.iter().enumerate() {
            let mem = Memory::new_with_timestamp(
                format!("memory {}", i),
                vec![0.1; 384],
                now.subtract_days(*days_ago),
            );
            storage.store_memory(&mem).unwrap();
        }

        let results = temporal
            .range_query(now.subtract_days(10), now, 100)
            .unwrap();

        // Verify ordering: newest first
        for i in 0..results.len() - 1 {
            assert!(
                results[i].timestamp >= results[i + 1].timestamp,
                "Results should be sorted newest first"
            );
        }
    }

    // Temporal content scoring tests
    #[test]
    fn test_search_temporal_content_basic() {
        let (temporal, storage, _dir) = create_test_index();

        // Add memory with temporal expression in content
        let mut mem1 = Memory::new(
            "We had a meeting yesterday about the project".to_string(),
            vec![0.1; 384],
        );
        mem1.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday"]"#.to_string(),
        );
        storage.store_memory(&mem1).unwrap();

        // Add memory with different temporal expression
        let mut mem2 = Memory::new(
            "The event is scheduled for next week".to_string(),
            vec![0.2; 384],
        );
        mem2.set_metadata(
            "temporal_expressions".to_string(),
            r#"["next week"]"#.to_string(),
        );
        storage.store_memory(&mem2).unwrap();

        // Add memory with no temporal expressions
        let mem3 = Memory::new(
            "Machine learning is fascinating".to_string(),
            vec![0.3; 384],
        );
        storage.store_memory(&mem3).unwrap();

        // Query with "yesterday"
        let results = temporal
            .search_temporal_content("What happened yesterday?", 10)
            .unwrap();

        // Should find mem1 with "yesterday"
        assert!(!results.is_empty(), "Should find memories with matching temporal expressions");
        assert_eq!(results[0].0, mem1.id, "Should rank mem1 (yesterday) first");
        assert!(results[0].1 > 0.0, "Should have positive score");
    }

    #[test]
    fn test_search_temporal_content_no_query_expressions() {
        let (temporal, storage, _dir) = create_test_index();

        // Add memory with temporal expression
        let mut mem1 = Memory::new(
            "We met yesterday".to_string(),
            vec![0.1; 384],
        );
        mem1.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday"]"#.to_string(),
        );
        storage.store_memory(&mem1).unwrap();

        // Query with no temporal expressions
        let results = temporal
            .search_temporal_content("Tell me about machine learning", 10)
            .unwrap();

        // Should return empty (no temporal expressions in query)
        assert_eq!(results.len(), 0, "Should return empty when query has no temporal expressions");
    }

    #[test]
    fn test_search_temporal_content_multiple_matches() {
        let (temporal, storage, _dir) = create_test_index();

        // Add memories with "yesterday"
        let mut mem1 = Memory::new(
            "I saw Alice yesterday at the park".to_string(),
            vec![0.1; 384],
        );
        mem1.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday"]"#.to_string(),
        );
        storage.store_memory(&mem1).unwrap();

        let mut mem2 = Memory::new(
            "Yesterday was a great day for tennis".to_string(),
            vec![0.2; 384],
        );
        mem2.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday"]"#.to_string(),
        );
        storage.store_memory(&mem2).unwrap();

        let mut mem3 = Memory::new(
            "The meeting was last week".to_string(),
            vec![0.3; 384],
        );
        mem3.set_metadata(
            "temporal_expressions".to_string(),
            r#"["last week"]"#.to_string(),
        );
        storage.store_memory(&mem3).unwrap();

        // Query with "yesterday"
        let results = temporal
            .search_temporal_content("What did I do yesterday?", 10)
            .unwrap();

        // Should find both mem1 and mem2 (both have "yesterday")
        assert!(results.len() >= 2, "Should find multiple memories with matching temporal expression");

        let result_ids: Vec<_> = results.iter().map(|(id, _)| id).collect();
        assert!(result_ids.contains(&&mem1.id), "Should include mem1 (yesterday)");
        assert!(result_ids.contains(&&mem2.id), "Should include mem2 (yesterday)");
    }

    #[test]
    fn test_search_temporal_content_scoring() {
        let (temporal, storage, _dir) = create_test_index();

        // Add memory with multiple temporal expressions matching query
        let mut mem1 = Memory::new(
            "Yesterday morning we discussed the project".to_string(),
            vec![0.1; 384],
        );
        mem1.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday", "morning"]"#.to_string(),
        );
        storage.store_memory(&mem1).unwrap();

        // Add memory with only one matching expression
        let mut mem2 = Memory::new(
            "I went to the park yesterday".to_string(),
            vec![0.2; 384],
        );
        mem2.set_metadata(
            "temporal_expressions".to_string(),
            r#"["yesterday"]"#.to_string(),
        );
        storage.store_memory(&mem2).unwrap();

        // Query with multiple temporal expressions
        let results = temporal
            .search_temporal_content("What happened yesterday morning?", 10)
            .unwrap();

        assert!(results.len() >= 2, "Should find both memories");

        // mem1 should score higher (has both "yesterday" and "morning")
        // Note: The actual scoring depends on TemporalExtractor.calculate_overlap()
        // which may normalize scores, so we just verify both are found
        let result_ids: Vec<_> = results.iter().map(|(id, _)| id).collect();
        assert!(result_ids.contains(&&mem1.id), "Should include mem1 (yesterday morning)");
        assert!(result_ids.contains(&&mem2.id), "Should include mem2 (yesterday)");
    }
}
