//! Storage engine implementation
//!
//! Wraps redb and provides CRUD operations for memories and indexes.

use crate::{
    types::{Memory, MemoryId, Timestamp},
    Error, Result,
};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::FileHeader;

// Table definitions
const MEMORIES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("memories");
pub(crate) const TEMPORAL_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("temporal_index");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");
const MEMORY_ID_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("memory_id_index");
const CAUSAL_GRAPH: TableDefinition<&str, &[u8]> = TableDefinition::new("causal_graph");

/// Storage engine wrapper around redb
///
/// Provides ACID transactions and persistent storage for all MnemeFusion data.
pub struct StorageEngine {
    db: Database,
    path: PathBuf,
}

impl StorageEngine {
    /// Open or create a database at the specified path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let db = Database::create(path)?;

        let mut engine = Self {
            db,
            path: path.to_path_buf(),
        };

        // Initialize tables
        engine.init_tables()?;

        // Store or validate header
        engine.init_header()?;

        Ok(engine)
    }

    /// Initialize required tables
    fn init_tables(&self) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let _ = write_txn.open_table(MEMORIES)?;
            let _ = write_txn.open_table(TEMPORAL_INDEX)?;
            let _ = write_txn.open_table(METADATA_TABLE)?;
            let _ = write_txn.open_table(MEMORY_ID_INDEX)?;
            let _ = write_txn.open_table(CAUSAL_GRAPH)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Initialize or validate file header
    fn init_header(&mut self) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;

            // Check if header exists
            let header_exists = table.get("file_header")?.is_some();

            if header_exists {
                // Read and validate existing header
                let existing = table.get("file_header")?.unwrap();
                let existing_bytes = existing.value().to_vec();
                let header = FileHeader::from_bytes(&existing_bytes)?;
                header.validate()?;
            } else {
                // Create new header
                let header = FileHeader::new();
                table.insert("file_header", header.to_bytes().as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Store a memory record
    pub fn store_memory(&self, memory: &Memory) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut memories = write_txn.open_table(MEMORIES)?;
            let mut temporal = write_txn.open_table(TEMPORAL_INDEX)?;
            let mut id_index = write_txn.open_table(MEMORY_ID_INDEX)?;

            // Serialize memory
            let memory_data = self.serialize_memory(memory)?;

            // Store memory
            memories.insert(memory.id.as_bytes().as_slice(), memory_data.as_slice())?;

            // Index by timestamp
            temporal.insert(memory.created_at.as_micros(), memory.id.as_bytes().as_slice())?;

            // Index by u64 (for vector index lookups)
            id_index.insert(memory.id.to_u64(), memory.id.as_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Retrieve a memory by ID
    pub fn get_memory(&self, id: &MemoryId) -> Result<Option<Memory>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;

        match table.get(id.as_bytes().as_slice())? {
            Some(data) => {
                let memory = self.deserialize_memory(data.value())?;
                Ok(Some(memory))
            }
            None => Ok(None),
        }
    }

    /// Retrieve a memory by its u64 key (used by vector index)
    pub fn get_memory_by_u64(&self, key: u64) -> Result<Option<Memory>> {
        let read_txn = self.db.begin_read()?;
        let id_index = read_txn.open_table(MEMORY_ID_INDEX)?;
        let memories = read_txn.open_table(MEMORIES)?;

        // First lookup the full MemoryId from the u64 index
        match id_index.get(key)? {
            Some(id_bytes) => {
                // Then fetch the memory using the full ID
                match memories.get(id_bytes.value())? {
                    Some(data) => {
                        let memory = self.deserialize_memory(data.value())?;
                        Ok(Some(memory))
                    }
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    /// Delete a memory by ID
    pub fn delete_memory(&self, id: &MemoryId) -> Result<bool> {
        let write_txn = self.db.begin_write()?;
        let removed = {
            let mut memories = write_txn.open_table(MEMORIES)?;
            let mut id_index = write_txn.open_table(MEMORY_ID_INDEX)?;

            let result = memories.remove(id.as_bytes().as_slice())?;

            // Also remove from ID index
            if result.is_some() {
                id_index.remove(id.to_u64())?;
            }

            result.is_some()
        };
        write_txn.commit()?;
        Ok(removed)
    }

    /// Get all memory IDs (for testing/debugging)
    pub fn list_memory_ids(&self) -> Result<Vec<MemoryId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;

        let mut ids = Vec::new();
        for item in table.iter()? {
            let (key, _) = item?;
            let id = MemoryId::from_bytes(key.value())?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Get memory count
    pub fn count_memories(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;
        Ok(table.len()? as usize)
    }

    /// Serialize memory to bytes
    fn serialize_memory(&self, memory: &Memory) -> Result<Vec<u8>> {
        // Simple serialization format:
        // [id (16 bytes)][timestamp (8 bytes)][content_len (4 bytes)][content][embedding_len (4 bytes)][embedding][metadata_len (4 bytes)][metadata]

        let mut bytes = Vec::new();

        // ID
        bytes.extend_from_slice(memory.id.as_bytes());

        // Timestamp
        bytes.extend_from_slice(&memory.created_at.to_bytes());

        // Content
        let content_bytes = memory.content.as_bytes();
        bytes.extend_from_slice(&(content_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(content_bytes);

        // Embedding
        bytes.extend_from_slice(&(memory.embedding.len() as u32).to_le_bytes());
        for val in &memory.embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Metadata
        let metadata_str = serde_json::to_string(&memory.metadata)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let metadata_bytes = metadata_str.as_bytes();
        bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(metadata_bytes);

        Ok(bytes)
    }

    /// Deserialize memory from bytes
    fn deserialize_memory(&self, bytes: &[u8]) -> Result<Memory> {
        let mut offset = 0;

        // ID
        if bytes.len() < offset + 16 {
            return Err(Error::Deserialization("Incomplete memory data".to_string()));
        }
        let id = MemoryId::from_bytes(&bytes[offset..offset + 16])?;
        offset += 16;

        // Timestamp
        if bytes.len() < offset + 8 {
            return Err(Error::Deserialization("Incomplete timestamp data".to_string()));
        }
        let created_at = Timestamp::from_bytes(&bytes[offset..offset + 8])?;
        offset += 8;

        // Content
        if bytes.len() < offset + 4 {
            return Err(Error::Deserialization("Incomplete content length".to_string()));
        }
        let content_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + content_len {
            return Err(Error::Deserialization("Incomplete content data".to_string()));
        }
        let content = String::from_utf8(bytes[offset..offset + content_len].to_vec())
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        offset += content_len;

        // Embedding
        if bytes.len() < offset + 4 {
            return Err(Error::Deserialization("Incomplete embedding length".to_string()));
        }
        let embedding_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + embedding_len * 4 {
            return Err(Error::Deserialization("Incomplete embedding data".to_string()));
        }
        let mut embedding = Vec::with_capacity(embedding_len);
        for _ in 0..embedding_len {
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            embedding.push(val);
            offset += 4;
        }

        // Metadata
        if bytes.len() < offset + 4 {
            return Err(Error::Deserialization("Incomplete metadata length".to_string()));
        }
        let metadata_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + metadata_len {
            return Err(Error::Deserialization("Incomplete metadata data".to_string()));
        }
        let metadata_str = String::from_utf8(bytes[offset..offset + metadata_len].to_vec())
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_str)
            .map_err(|e| Error::Deserialization(e.to_string()))?;

        Ok(Memory {
            id,
            content,
            embedding,
            created_at,
            metadata,
        })
    }

    /// Get the database path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get a reference to the underlying database
    ///
    /// This is used by index implementations (TemporalIndex, etc.) to access
    /// tables directly for specialized queries.
    pub(crate) fn db(&self) -> &Database {
        &self.db
    }

    /// Store vector index data
    pub fn store_vector_index(&self, buffer: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert("vector_index", buffer)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load vector index data
    pub fn load_vector_index(&self) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;

        match table.get("vector_index")? {
            Some(data) => Ok(Some(data.value().to_vec())),
            None => Ok(None),
        }
    }

    /// Store causal graph data
    pub fn store_causal_graph(&self, data: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CAUSAL_GRAPH)?;
            table.insert("graph", data)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load causal graph data
    pub fn load_causal_graph(&self) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CAUSAL_GRAPH)?;

        match table.get("graph")? {
            Some(data) => Ok(Some(data.value().to_vec())),
            None => Ok(None),
        }
    }
}

// Add serde_json for metadata serialization
// This is a temporary solution - we'll use rkyv for zero-copy in later sprints

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_storage_engine_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let engine = StorageEngine::open(&path).unwrap();
        assert_eq!(engine.path(), path);
    }

    #[test]
    fn test_storage_engine_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let memory = Memory::new("test content".to_string(), vec![0.1, 0.2, 0.3]);
        let id = memory.id.clone();

        engine.store_memory(&memory).unwrap();

        let retrieved = engine.get_memory(&id).unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.content, "test content");
        assert_eq!(retrieved.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_storage_engine_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let memory = Memory::new("test".to_string(), vec![0.1]);
        let id = memory.id.clone();

        engine.store_memory(&memory).unwrap();
        assert!(engine.get_memory(&id).unwrap().is_some());

        let deleted = engine.delete_memory(&id).unwrap();
        assert!(deleted);
        assert!(engine.get_memory(&id).unwrap().is_none());
    }

    #[test]
    fn test_storage_engine_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let id = MemoryId::new();
        assert!(engine.get_memory(&id).unwrap().is_none());
    }

    #[test]
    fn test_storage_engine_multiple_memories() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mem1 = Memory::new("first".to_string(), vec![0.1]);
        let mem2 = Memory::new("second".to_string(), vec![0.2]);
        let mem3 = Memory::new("third".to_string(), vec![0.3]);

        engine.store_memory(&mem1).unwrap();
        engine.store_memory(&mem2).unwrap();
        engine.store_memory(&mem3).unwrap();

        assert_eq!(engine.count_memories().unwrap(), 3);

        let ids = engine.list_memory_ids().unwrap();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_storage_engine_with_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());
        metadata.insert("category".to_string(), "example".to_string());

        let memory = Memory::new_with_metadata("test".to_string(), vec![0.1], metadata);
        let id = memory.id.clone();

        engine.store_memory(&memory).unwrap();

        let retrieved = engine.get_memory(&id).unwrap().unwrap();
        assert_eq!(retrieved.metadata.len(), 2);
        assert_eq!(retrieved.metadata.get("source"), Some(&"test".to_string()));
        assert_eq!(retrieved.metadata.get("category"), Some(&"example".to_string()));
    }

    #[test]
    fn test_storage_engine_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let memory = Memory::new("persistent".to_string(), vec![0.5]);
        let id = memory.id.clone();

        // Store in first instance
        {
            let engine = StorageEngine::open(&path).unwrap();
            engine.store_memory(&memory).unwrap();
        }

        // Retrieve in second instance
        {
            let engine = StorageEngine::open(&path).unwrap();
            let retrieved = engine.get_memory(&id).unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().content, "persistent");
        }
    }
}
