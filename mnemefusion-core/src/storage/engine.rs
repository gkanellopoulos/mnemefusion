//! Storage engine implementation
//!
//! Wraps redb and provides CRUD operations for memories and indexes.

use crate::{
    types::{Entity, EntityId, Memory, MemoryId, Timestamp},
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
const ENTITIES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("entities");
const ENTITY_NAMES: TableDefinition<&str, &[u8]> = TableDefinition::new("entity_names");
const CONTENT_HASH_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("content_hash_index");
const LOGICAL_KEY_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("logical_key_index");

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
            let _ = write_txn.open_table(ENTITIES)?;
            let _ = write_txn.open_table(ENTITY_NAMES)?;
            let _ = write_txn.open_table(CONTENT_HASH_INDEX)?;
            let _ = write_txn.open_table(LOGICAL_KEY_INDEX)?;
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

    /// Store an entity
    pub fn store_entity(&self, entity: &Entity) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut entities = write_txn.open_table(ENTITIES)?;
            let mut names = write_txn.open_table(ENTITY_NAMES)?;

            // Serialize entity to JSON
            let entity_data = serde_json::to_vec(entity)
                .map_err(|e| Error::Serialization(e.to_string()))?;

            // Store entity by ID
            entities.insert(entity.id.as_bytes().as_slice(), entity_data.as_slice())?;

            // Index by normalized name (case-insensitive)
            let normalized_name = entity.normalized_name();
            names.insert(normalized_name.as_str(), entity.id.as_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get an entity by ID
    pub fn get_entity(&self, id: &EntityId) -> Result<Option<Entity>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITIES)?;

        match table.get(id.as_bytes().as_slice())? {
            Some(data) => {
                let entity: Entity = serde_json::from_slice(data.value())
                    .map_err(|e| Error::Deserialization(e.to_string()))?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Find an entity by name (case-insensitive)
    pub fn find_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        let read_txn = self.db.begin_read()?;
        let names_table = read_txn.open_table(ENTITY_NAMES)?;
        let entities_table = read_txn.open_table(ENTITIES)?;

        // Normalize the search name
        let normalized = name.to_lowercase();

        // Look up entity ID by normalized name
        match names_table.get(normalized.as_str())? {
            Some(id_bytes) => {
                let id_bytes = id_bytes.value().to_vec();
                let entity_id = EntityId::from_bytes(&id_bytes)?;

                // Get the entity
                match entities_table.get(entity_id.as_bytes().as_slice())? {
                    Some(data) => {
                        let entity: Entity = serde_json::from_slice(data.value())
                            .map_err(|e| Error::Deserialization(e.to_string()))?;
                        Ok(Some(entity))
                    }
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    /// Delete an entity
    pub fn delete_entity(&self, id: &EntityId) -> Result<bool> {
        let write_txn = self.db.begin_write()?;
        let deleted = {
            let mut entities = write_txn.open_table(ENTITIES)?;
            let mut names = write_txn.open_table(ENTITY_NAMES)?;

            // Get entity to find its name for name index cleanup
            // Store the normalized name before dropping the guard
            let normalized_name = if let Some(data) = entities.get(id.as_bytes().as_slice())? {
                let entity: Entity = serde_json::from_slice(data.value())
                    .map_err(|e| Error::Deserialization(e.to_string()))?;
                Some(entity.normalized_name())
            } else {
                None
            };

            // Now we can mutate
            if let Some(name) = normalized_name {
                // Remove from name index
                names.remove(name.as_str())?;

                // Remove entity
                entities.remove(id.as_bytes().as_slice())?;
                true
            } else {
                false
            }
        };
        write_txn.commit()?;
        Ok(deleted)
    }

    /// List all entities
    pub fn list_entities(&self) -> Result<Vec<Entity>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITIES)?;

        let mut entities = Vec::new();
        for result in table.iter()? {
            let (_, value) = result?;
            let entity: Entity = serde_json::from_slice(value.value())
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            entities.push(entity);
        }

        Ok(entities)
    }

    /// Count entities
    pub fn count_entities(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITIES)?;
        Ok(table.len()? as usize)
    }

    /// Store entity graph data
    pub fn store_entity_graph(&self, data: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert("entity_graph", data)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load entity graph data
    pub fn load_entity_graph(&self) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;

        match table.get("entity_graph")? {
            Some(data) => Ok(Some(data.value().to_vec())),
            None => Ok(None),
        }
    }

    // Content hash operations for deduplication

    /// Store content hash → memory ID mapping
    ///
    /// Used for deduplication to quickly find if content already exists
    pub fn store_content_hash(&self, hash: &str, memory_id: &MemoryId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CONTENT_HASH_INDEX)?;
            table.insert(hash, memory_id.as_bytes() as &[u8])?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Find memory ID by content hash
    ///
    /// Returns None if hash not found (content is unique)
    pub fn find_by_content_hash(&self, hash: &str) -> Result<Option<MemoryId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CONTENT_HASH_INDEX)?;

        match table.get(hash)? {
            Some(bytes) => {
                let id = MemoryId::from_bytes(bytes.value())?;
                Ok(Some(id))
            }
            None => Ok(None),
        }
    }

    /// Delete content hash mapping
    pub fn delete_content_hash(&self, hash: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CONTENT_HASH_INDEX)?;
            table.remove(hash)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    // Logical key operations for upsert

    /// Store logical key → memory ID mapping
    ///
    /// Used for upsert operations with developer-defined keys
    pub fn store_logical_key(&self, key: &str, memory_id: &MemoryId) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(LOGICAL_KEY_INDEX)?;
            table.insert(key, memory_id.as_bytes() as &[u8])?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Find memory ID by logical key
    ///
    /// Returns None if key not found
    pub fn find_by_logical_key(&self, key: &str) -> Result<Option<MemoryId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(LOGICAL_KEY_INDEX)?;

        match table.get(key)? {
            Some(bytes) => {
                let id = MemoryId::from_bytes(bytes.value())?;
                Ok(Some(id))
            }
            None => Ok(None),
        }
    }

    /// Delete logical key mapping
    pub fn delete_logical_key(&self, key: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(LOGICAL_KEY_INDEX)?;
            table.remove(key)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Update logical key mapping to point to a new memory ID
    ///
    /// This is used when an upsert operation replaces an existing memory
    pub fn update_logical_key(&self, key: &str, new_memory_id: &MemoryId) -> Result<()> {
        // Just overwrite - same as store
        self.store_logical_key(key, new_memory_id)
    }

    // Namespace operations

    /// List all namespaces in the database
    ///
    /// Scans all memories and extracts unique namespace values.
    /// Returns sorted list of namespace strings (excluding default namespace "").
    ///
    /// # Performance
    ///
    /// O(n) where n = total memories. Can be cached if needed.
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;

        let mut namespaces = std::collections::HashSet::new();

        // Scan all memories
        for entry in table.iter()? {
            let (_, value) = entry?;
            let memory_data = value.value();

            // Deserialize memory to get namespace
            if let Ok(memory) = self.deserialize_memory(memory_data) {
                let ns = memory.get_namespace();
                if !ns.is_empty() {
                    namespaces.insert(ns);
                }
            }
        }

        let mut result: Vec<String> = namespaces.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Count memories in a specific namespace
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to count (empty string "" for default)
    ///
    /// # Returns
    ///
    /// Number of memories in the namespace
    pub fn count_namespace(&self, namespace: &str) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;

        let mut count = 0;

        // Scan all memories
        for entry in table.iter()? {
            let (_, value) = entry?;
            let memory_data = value.value();

            // Deserialize memory to check namespace
            if let Ok(memory) = self.deserialize_memory(memory_data) {
                if memory.get_namespace() == namespace {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// List all memory IDs in a specific namespace
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace to list (empty string "" for default)
    ///
    /// # Returns
    ///
    /// Vector of MemoryIds in the namespace
    pub fn list_namespace_ids(&self, namespace: &str) -> Result<Vec<MemoryId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MEMORIES)?;

        let mut ids = Vec::new();

        // Scan all memories
        for entry in table.iter()? {
            let (key, value) = entry?;
            let memory_data = value.value();

            // Deserialize memory to check namespace
            if let Ok(memory) = self.deserialize_memory(memory_data) {
                if memory.get_namespace() == namespace {
                    // Get memory ID from key
                    let id = MemoryId::from_bytes(key.value())?;
                    ids.push(id);
                }
            }
        }

        Ok(ids)
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

    #[test]
    fn test_storage_list_namespaces_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // No namespaces initially
        let namespaces = engine.list_namespaces().unwrap();
        assert!(namespaces.is_empty());
    }

    #[test]
    fn test_storage_list_namespaces() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Add memories to different namespaces
        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.set_namespace("user_123");
        engine.store_memory(&mem1).unwrap();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.set_namespace("user_456");
        engine.store_memory(&mem2).unwrap();

        let mut mem3 = Memory::new("content 3".to_string(), vec![0.3; 384]);
        mem3.set_namespace("user_123"); // Duplicate namespace
        engine.store_memory(&mem3).unwrap();

        // Default namespace (no set_namespace)
        let mem4 = Memory::new("content 4".to_string(), vec![0.4; 384]);
        engine.store_memory(&mem4).unwrap();

        // Should return 2 unique non-default namespaces (sorted)
        let namespaces = engine.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);
        assert_eq!(namespaces[0], "user_123");
        assert_eq!(namespaces[1], "user_456");
    }

    #[test]
    fn test_storage_count_namespace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Add memories to different namespaces
        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.set_namespace("user_123");
        engine.store_memory(&mem1).unwrap();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.set_namespace("user_123");
        engine.store_memory(&mem2).unwrap();

        let mut mem3 = Memory::new("content 3".to_string(), vec![0.3; 384]);
        mem3.set_namespace("user_456");
        engine.store_memory(&mem3).unwrap();

        let mem4 = Memory::new("content 4".to_string(), vec![0.4; 384]);
        engine.store_memory(&mem4).unwrap();

        // Count by namespace
        assert_eq!(engine.count_namespace("user_123").unwrap(), 2);
        assert_eq!(engine.count_namespace("user_456").unwrap(), 1);
        assert_eq!(engine.count_namespace("").unwrap(), 1); // Default namespace
        assert_eq!(engine.count_namespace("nonexistent").unwrap(), 0);
    }

    #[test]
    fn test_storage_list_namespace_ids() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Add memories to namespace
        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.set_namespace("user_123");
        let id1 = mem1.id.clone();
        engine.store_memory(&mem1).unwrap();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.set_namespace("user_123");
        let id2 = mem2.id.clone();
        engine.store_memory(&mem2).unwrap();

        let mut mem3 = Memory::new("content 3".to_string(), vec![0.3; 384]);
        mem3.set_namespace("user_456");
        engine.store_memory(&mem3).unwrap();

        // List IDs by namespace
        let ids = engine.list_namespace_ids("user_123").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));

        let ids_456 = engine.list_namespace_ids("user_456").unwrap();
        assert_eq!(ids_456.len(), 1);

        let ids_empty = engine.list_namespace_ids("nonexistent").unwrap();
        assert!(ids_empty.is_empty());
    }

    #[test]
    fn test_storage_namespace_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Memory without explicit namespace
        let mem = Memory::new("default content".to_string(), vec![0.1; 384]);
        let id = mem.id.clone();
        engine.store_memory(&mem).unwrap();

        // Should be in default namespace ""
        assert_eq!(engine.count_namespace("").unwrap(), 1);

        let ids = engine.list_namespace_ids("").unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id);

        // Should not appear in list_namespaces (only non-default)
        let namespaces = engine.list_namespaces().unwrap();
        assert!(namespaces.is_empty());
    }
}
