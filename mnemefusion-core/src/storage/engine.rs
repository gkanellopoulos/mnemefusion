//! Storage engine implementation
//!
//! Wraps redb and provides CRUD operations for memories and indexes.

use crate::{
    types::{Entity, EntityId, EntityProfile, Memory, MemoryId, Timestamp},
    Error, Result,
};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::FileHeader;

// Table definitions
const MEMORIES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("memories");
pub(crate) const TEMPORAL_INDEX: TableDefinition<u64, &[u8]> =
    TableDefinition::new("temporal_index");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");
const MEMORY_ID_INDEX: TableDefinition<u64, &[u8]> = TableDefinition::new("memory_id_index");
const CAUSAL_GRAPH: TableDefinition<&str, &[u8]> = TableDefinition::new("causal_graph");
const ENTITIES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("entities");
const ENTITY_NAMES: TableDefinition<&str, &[u8]> = TableDefinition::new("entity_names");
const CONTENT_HASH_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("content_hash_index");
const LOGICAL_KEY_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("logical_key_index");
const METADATA_INDEX: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata_index");
const ENTITY_PROFILES: TableDefinition<&str, &[u8]> = TableDefinition::new("entity_profiles");

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

        // Check if file exists and validate minimum size
        if path.exists() {
            let metadata = std::fs::metadata(path)?;
            let file_size = metadata.len();

            // Minimum valid file size (redb has minimum overhead + 64 byte header)
            // redb files are typically at least a few KB even when empty
            const MIN_FILE_SIZE: u64 = 512; // Conservative minimum

            if file_size < MIN_FILE_SIZE {
                return Err(Error::FileTruncated(format!(
                    "File size ({} bytes) is too small to be a valid database",
                    file_size
                )));
            }
        }

        let db = Database::create(path)?;

        let mut engine = Self {
            db,
            path: path.to_path_buf(),
        };

        // Initialize tables
        engine.init_tables()?;

        // Store or validate header
        engine.init_header()?;

        // Validate database integrity (for existing databases)
        if path.exists() {
            engine.validate_database()?;
        }

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
            let _ = write_txn.open_table(METADATA_INDEX)?;
            let _ = write_txn.open_table(ENTITY_PROFILES)?;
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

    /// Validate database integrity
    ///
    /// Verifies that all required tables exist and are accessible.
    /// This helps detect corrupted or incomplete database files.
    fn validate_database(&self) -> Result<()> {
        let read_txn = self.db.begin_read()?;

        // Check that all required tables exist and are accessible
        // We open each table individually since they have different type parameters

        if let Err(e) = read_txn.open_table(MEMORIES) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'memories' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(TEMPORAL_INDEX) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'temporal_index' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(METADATA_TABLE) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'metadata' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(MEMORY_ID_INDEX) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'memory_id_index' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(CAUSAL_GRAPH) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'causal_graph' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(ENTITIES) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'entities' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(ENTITY_NAMES) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'entity_names' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(CONTENT_HASH_INDEX) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'content_hash_index' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(LOGICAL_KEY_INDEX) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'logical_key_index' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(METADATA_INDEX) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'metadata_index' is missing or corrupt: {}",
                e
            )));
        }

        if let Err(e) = read_txn.open_table(ENTITY_PROFILES) {
            return Err(Error::DatabaseCorruption(format!(
                "Required table 'entity_profiles' is missing or corrupt: {}",
                e
            )));
        }

        // Validate header exists
        let metadata = read_txn.open_table(METADATA_TABLE)?;
        match metadata.get("file_header")? {
            Some(header_bytes) => {
                // Validate header format
                let header = FileHeader::from_bytes(header_bytes.value())?;
                header.validate()?;
            }
            None => {
                return Err(Error::DatabaseCorruption(
                    "File header is missing".to_string(),
                ));
            }
        }

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
            temporal.insert(
                memory.created_at.as_micros(),
                memory.id.as_bytes().as_slice(),
            )?;

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
            return Err(Error::Deserialization(
                "Incomplete timestamp data".to_string(),
            ));
        }
        let created_at = Timestamp::from_bytes(&bytes[offset..offset + 8])?;
        offset += 8;

        // Content
        if bytes.len() < offset + 4 {
            return Err(Error::Deserialization(
                "Incomplete content length".to_string(),
            ));
        }
        let content_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + content_len {
            return Err(Error::Deserialization(
                "Incomplete content data".to_string(),
            ));
        }
        let content = String::from_utf8(bytes[offset..offset + content_len].to_vec())
            .map_err(|e| Error::Deserialization(e.to_string()))?;
        offset += content_len;

        // Embedding
        if bytes.len() < offset + 4 {
            return Err(Error::Deserialization(
                "Incomplete embedding length".to_string(),
            ));
        }
        let embedding_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + embedding_len * 4 {
            return Err(Error::Deserialization(
                "Incomplete embedding data".to_string(),
            ));
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
            return Err(Error::Deserialization(
                "Incomplete metadata length".to_string(),
            ));
        }
        let metadata_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if bytes.len() < offset + metadata_len {
            return Err(Error::Deserialization(
                "Incomplete metadata data".to_string(),
            ));
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

    /// Store BM25 index data
    pub fn store_bm25_index(&self, buffer: &[u8]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert("bm25_index", buffer)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load BM25 index data
    pub fn load_bm25_index(&self) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;

        match table.get("bm25_index")? {
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
            let entity_data =
                serde_json::to_vec(entity).map_err(|e| Error::Serialization(e.to_string()))?;

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

    // Metadata index operations

    /// Build a metadata index key
    ///
    /// Format: "{field}:{value}:{namespace}"
    fn metadata_index_key(field: &str, value: &str, namespace: &str) -> String {
        format!("{}:{}:{}", field, value, namespace)
    }

    /// Store a metadata field value in the index
    ///
    /// Associates a memory with a specific metadata field value in a namespace.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name
    /// * `value` - The field value
    /// * `namespace` - The namespace the memory belongs to
    /// * `memory_id` - The memory ID to associate with this field value
    pub fn add_to_metadata_index(
        &self,
        field: &str,
        value: &str,
        namespace: &str,
        memory_id: &MemoryId,
    ) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_INDEX)?;

            let key = Self::metadata_index_key(field, value, namespace);

            // Get existing memory IDs for this key
            let mut ids: Vec<MemoryId> = match table.get(key.as_str())? {
                Some(data) => serde_json::from_slice(data.value())?,
                None => Vec::new(),
            };

            // Add new ID if not already present
            if !ids.contains(memory_id) {
                ids.push(memory_id.clone());

                // Serialize and store
                let data = serde_json::to_vec(&ids)?;
                table.insert(key.as_str(), data.as_slice())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Remove a memory from a metadata index entry
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name
    /// * `value` - The field value
    /// * `namespace` - The namespace
    /// * `memory_id` - The memory ID to remove
    pub fn remove_from_metadata_index(
        &self,
        field: &str,
        value: &str,
        namespace: &str,
        memory_id: &MemoryId,
    ) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_INDEX)?;

            let key = Self::metadata_index_key(field, value, namespace);

            // Get existing memory IDs for this key and clone the data
            let ids_data = table.get(key.as_str())?.map(|data| data.value().to_vec());

            if let Some(data_vec) = ids_data {
                let mut ids: Vec<MemoryId> = serde_json::from_slice(&data_vec)?;

                // Remove the memory ID
                ids.retain(|id| id != memory_id);

                if ids.is_empty() {
                    // Remove the index entry if no more memories
                    table.remove(key.as_str())?;
                } else {
                    // Update with remaining IDs
                    let data = serde_json::to_vec(&ids)?;
                    table.insert(key.as_str(), data.as_slice())?;
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Find all memory IDs matching a metadata field value
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name
    /// * `value` - The field value to match
    /// * `namespace` - The namespace to search in
    ///
    /// # Returns
    ///
    /// Vector of memory IDs that have the specified metadata field value
    pub fn find_by_metadata(
        &self,
        field: &str,
        value: &str,
        namespace: &str,
    ) -> Result<Vec<MemoryId>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(METADATA_INDEX)?;

        let key = Self::metadata_index_key(field, value, namespace);

        match table.get(key.as_str())? {
            Some(data) => {
                let ids: Vec<MemoryId> = serde_json::from_slice(data.value())?;
                Ok(ids)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Remove all metadata index entries for a memory
    ///
    /// Used when deleting a memory to clean up all its metadata indexes.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory being deleted
    /// * `indexed_fields` - List of fields that are indexed
    pub fn remove_metadata_indexes_for_memory(
        &self,
        memory: &Memory,
        indexed_fields: &[String],
    ) -> Result<()> {
        let namespace = memory.get_namespace();

        // For each indexed field, remove this memory from the index
        for field in indexed_fields {
            if let Some(value) = memory.metadata.get(field) {
                self.remove_from_metadata_index(field, value, &namespace, &memory.id)?;
            }
        }

        Ok(())
    }

    // Entity Profile operations

    /// Store an entity profile
    ///
    /// Profiles are keyed by the normalized (lowercase) entity name for
    /// case-insensitive lookups.
    ///
    /// # Arguments
    ///
    /// * `profile` - The entity profile to store
    ///
    /// # Example
    ///
    /// ```ignore
    /// let profile = EntityProfile::new(EntityId::new(), "Alice".into(), "person".into());
    /// storage.store_entity_profile(&profile)?;
    /// ```
    pub fn store_entity_profile(&self, profile: &EntityProfile) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(ENTITY_PROFILES)?;
            let key = profile.name.to_lowercase();
            let data =
                serde_json::to_vec(profile).map_err(|e| Error::Serialization(e.to_string()))?;
            table.insert(key.as_str(), data.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get an entity profile by name (case-insensitive)
    ///
    /// # Arguments
    ///
    /// * `name` - The entity name to look up
    ///
    /// # Returns
    ///
    /// The entity profile if found, or None
    ///
    /// # Example
    ///
    /// ```ignore
    /// let profile = storage.get_entity_profile("Alice")?;
    /// if let Some(p) = profile {
    ///     println!("Found profile for {}", p.name);
    /// }
    /// ```
    pub fn get_entity_profile(&self, name: &str) -> Result<Option<EntityProfile>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITY_PROFILES)?;
        let key = name.to_lowercase();

        match table.get(key.as_str())? {
            Some(data) => {
                let profile: EntityProfile = serde_json::from_slice(data.value())
                    .map_err(|e| Error::Deserialization(e.to_string()))?;
                Ok(Some(profile))
            }
            None => Ok(None),
        }
    }

    /// List all entity profiles
    ///
    /// # Returns
    ///
    /// Vector of all entity profiles in the database
    ///
    /// # Performance
    ///
    /// O(n) where n = number of profiles. Use with caution on large databases.
    pub fn list_entity_profiles(&self) -> Result<Vec<EntityProfile>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITY_PROFILES)?;

        let mut profiles = Vec::new();
        for result in table.iter()? {
            let (_, value) = result?;
            let profile: EntityProfile = serde_json::from_slice(value.value())
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            profiles.push(profile);
        }
        Ok(profiles)
    }

    /// Delete an entity profile by name (case-insensitive)
    ///
    /// # Arguments
    ///
    /// * `name` - The entity name to delete
    ///
    /// # Returns
    ///
    /// true if the profile was deleted, false if it didn't exist
    pub fn delete_entity_profile(&self, name: &str) -> Result<bool> {
        let write_txn = self.db.begin_write()?;
        let deleted = {
            let mut table = write_txn.open_table(ENTITY_PROFILES)?;
            let key = name.to_lowercase();
            let result = table.remove(key.as_str())?;
            result.is_some()
        };
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Count entity profiles
    ///
    /// # Returns
    ///
    /// Number of entity profiles in the database
    pub fn count_entity_profiles(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(ENTITY_PROFILES)?;
        Ok(table.len()? as usize)
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
        assert_eq!(
            retrieved.metadata.get("category"),
            Some(&"example".to_string())
        );
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

    #[test]
    fn test_metadata_index_add_and_find() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem = Memory::new("test content".to_string(), vec![0.1; 384]);
        mem.metadata.insert("type".to_string(), "event".to_string());
        mem.metadata
            .insert("priority".to_string(), "high".to_string());
        let id = mem.id.clone();

        // Add to metadata index
        engine
            .add_to_metadata_index("type", "event", "", &id)
            .unwrap();
        engine
            .add_to_metadata_index("priority", "high", "", &id)
            .unwrap();

        // Find by metadata
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id);

        let ids = engine.find_by_metadata("priority", "high", "").unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id);

        // Non-existent value
        let ids = engine.find_by_metadata("type", "task", "").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_metadata_index_multiple_memories() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        let id1 = mem1.id.clone();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        let id2 = mem2.id.clone();

        let mut mem3 = Memory::new("content 3".to_string(), vec![0.3; 384]);
        mem3.metadata.insert("type".to_string(), "task".to_string());
        let id3 = mem3.id.clone();

        // Add to index
        engine
            .add_to_metadata_index("type", "event", "", &id1)
            .unwrap();
        engine
            .add_to_metadata_index("type", "event", "", &id2)
            .unwrap();
        engine
            .add_to_metadata_index("type", "task", "", &id3)
            .unwrap();

        // Find all events
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));

        // Find all tasks
        let ids = engine.find_by_metadata("type", "task", "").unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id3);
    }

    #[test]
    fn test_metadata_index_with_namespace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.set_namespace("user_123");
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        let id1 = mem1.id.clone();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.set_namespace("user_456");
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        let id2 = mem2.id.clone();

        // Add to index
        engine
            .add_to_metadata_index("type", "event", "user_123", &id1)
            .unwrap();
        engine
            .add_to_metadata_index("type", "event", "user_456", &id2)
            .unwrap();

        // Find in each namespace
        let ids = engine
            .find_by_metadata("type", "event", "user_123")
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id1);

        let ids = engine
            .find_by_metadata("type", "event", "user_456")
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id2);

        // Not found in wrong namespace
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_metadata_index_remove() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem = Memory::new("test content".to_string(), vec![0.1; 384]);
        mem.metadata.insert("type".to_string(), "event".to_string());
        let id = mem.id.clone();

        // Add to index
        engine
            .add_to_metadata_index("type", "event", "", &id)
            .unwrap();

        // Verify it's there
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert_eq!(ids.len(), 1);

        // Remove from index
        engine
            .remove_from_metadata_index("type", "event", "", &id)
            .unwrap();

        // Verify it's gone
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_metadata_index_remove_one_of_many() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem1 = Memory::new("content 1".to_string(), vec![0.1; 384]);
        mem1.metadata
            .insert("type".to_string(), "event".to_string());
        let id1 = mem1.id.clone();

        let mut mem2 = Memory::new("content 2".to_string(), vec![0.2; 384]);
        mem2.metadata
            .insert("type".to_string(), "event".to_string());
        let id2 = mem2.id.clone();

        // Add both to index
        engine
            .add_to_metadata_index("type", "event", "", &id1)
            .unwrap();
        engine
            .add_to_metadata_index("type", "event", "", &id2)
            .unwrap();

        // Verify both are there
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert_eq!(ids.len(), 2);

        // Remove one
        engine
            .remove_from_metadata_index("type", "event", "", &id1)
            .unwrap();

        // Verify only one remains
        let ids = engine.find_by_metadata("type", "event", "").unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id2);
    }

    #[test]
    fn test_metadata_index_remove_all_for_memory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let mut mem = Memory::new("test content".to_string(), vec![0.1; 384]);
        mem.metadata.insert("type".to_string(), "event".to_string());
        mem.metadata
            .insert("priority".to_string(), "high".to_string());
        mem.metadata
            .insert("category".to_string(), "work".to_string());
        let id = mem.id.clone();

        // Add to indexes
        engine
            .add_to_metadata_index("type", "event", "", &id)
            .unwrap();
        engine
            .add_to_metadata_index("priority", "high", "", &id)
            .unwrap();
        engine
            .add_to_metadata_index("category", "work", "", &id)
            .unwrap();

        // Verify all are indexed
        assert!(!engine
            .find_by_metadata("type", "event", "")
            .unwrap()
            .is_empty());
        assert!(!engine
            .find_by_metadata("priority", "high", "")
            .unwrap()
            .is_empty());
        assert!(!engine
            .find_by_metadata("category", "work", "")
            .unwrap()
            .is_empty());

        // Remove all indexes for this memory
        let indexed_fields = vec![
            "type".to_string(),
            "priority".to_string(),
            "category".to_string(),
        ];
        engine
            .remove_metadata_indexes_for_memory(&mem, &indexed_fields)
            .unwrap();

        // Verify all are removed
        assert!(engine
            .find_by_metadata("type", "event", "")
            .unwrap()
            .is_empty());
        assert!(engine
            .find_by_metadata("priority", "high", "")
            .unwrap()
            .is_empty());
        assert!(engine
            .find_by_metadata("category", "work", "")
            .unwrap()
            .is_empty());
    }

    // Validation tests for Task 4: File header validation on open

    #[test]
    fn test_truncated_file_detection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.mfdb");

        // Create a tiny truncated file
        std::fs::write(&path, b"MF").unwrap();

        // Should fail with FileTruncated error
        let result = StorageEngine::open(&path);
        assert!(result.is_err());

        if let Err(err) = result {
            assert!(matches!(err, Error::FileTruncated(_)));
        }
    }

    #[test]
    fn test_validate_database_integrity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        // Create a valid database
        let engine = StorageEngine::open(&path).unwrap();

        // Validation should pass
        assert!(engine.validate_database().is_ok());
    }

    #[test]
    fn test_validate_database_with_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        // Create database with some data
        let engine = StorageEngine::open(&path).unwrap();
        let mem = Memory::new("test content".to_string(), vec![0.1; 384]);
        engine.store_memory(&mem).unwrap();

        // Validation should still pass
        assert!(engine.validate_database().is_ok());

        // Close and reopen
        drop(engine);
        let engine = StorageEngine::open(&path).unwrap();

        // Validation should pass on reopen
        assert!(engine.validate_database().is_ok());
    }

    #[test]
    fn test_open_validates_existing_database() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        // Create a valid database
        {
            let _engine = StorageEngine::open(&path).unwrap();
        }

        // Reopening should validate the database
        let engine = StorageEngine::open(&path).unwrap();

        // Verify header is valid
        assert!(engine.validate_database().is_ok());
    }

    // Entity Profile tests

    #[test]
    fn test_entity_profile_store_and_retrieve() {
        use crate::types::{EntityFact, EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Create a profile
        let entity_id = EntityId::new();
        let mut profile = EntityProfile::new(
            entity_id.clone(),
            "Alice".to_string(),
            "person".to_string(),
        );

        // Add a fact
        let memory_id = MemoryId::new();
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            memory_id.clone(),
        ));
        profile.add_source_memory(memory_id);

        // Store
        engine.store_entity_profile(&profile).unwrap();

        // Retrieve
        let retrieved = engine.get_entity_profile("Alice").unwrap();
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "Alice");
        assert_eq!(retrieved.entity_type, "person");
        assert_eq!(retrieved.facts.get("occupation").unwrap().len(), 1);
        assert_eq!(retrieved.facts.get("occupation").unwrap()[0].value, "engineer");
    }

    #[test]
    fn test_entity_profile_case_insensitive_lookup() {
        use crate::types::{EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        engine.store_entity_profile(&profile).unwrap();

        // Case-insensitive lookup
        assert!(engine.get_entity_profile("alice").unwrap().is_some());
        assert!(engine.get_entity_profile("ALICE").unwrap().is_some());
        assert!(engine.get_entity_profile("Alice").unwrap().is_some());
        assert!(engine.get_entity_profile("aLiCe").unwrap().is_some());
    }

    #[test]
    fn test_entity_profile_not_found() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let result = engine.get_entity_profile("Nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_entity_profile_list() {
        use crate::types::{EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Create and store profiles
        let profile1 = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        let profile2 = EntityProfile::new(
            EntityId::new(),
            "Bob".to_string(),
            "person".to_string(),
        );
        let profile3 = EntityProfile::new(
            EntityId::new(),
            "Acme Corp".to_string(),
            "organization".to_string(),
        );

        engine.store_entity_profile(&profile1).unwrap();
        engine.store_entity_profile(&profile2).unwrap();
        engine.store_entity_profile(&profile3).unwrap();

        // List all
        let profiles = engine.list_entity_profiles().unwrap();
        assert_eq!(profiles.len(), 3);

        let names: Vec<_> = profiles.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        assert!(names.contains(&"Acme Corp"));
    }

    #[test]
    fn test_entity_profile_delete() {
        use crate::types::{EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        let profile = EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        );
        engine.store_entity_profile(&profile).unwrap();

        // Verify it exists
        assert!(engine.get_entity_profile("Alice").unwrap().is_some());

        // Delete
        let deleted = engine.delete_entity_profile("Alice").unwrap();
        assert!(deleted);

        // Verify it's gone
        assert!(engine.get_entity_profile("Alice").unwrap().is_none());

        // Deleting non-existent returns false
        let deleted = engine.delete_entity_profile("Alice").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_entity_profile_count() {
        use crate::types::{EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        assert_eq!(engine.count_entity_profiles().unwrap(), 0);

        engine.store_entity_profile(&EntityProfile::new(
            EntityId::new(),
            "Alice".to_string(),
            "person".to_string(),
        )).unwrap();
        assert_eq!(engine.count_entity_profiles().unwrap(), 1);

        engine.store_entity_profile(&EntityProfile::new(
            EntityId::new(),
            "Bob".to_string(),
            "person".to_string(),
        )).unwrap();
        assert_eq!(engine.count_entity_profiles().unwrap(), 2);
    }

    #[test]
    fn test_entity_profile_update() {
        use crate::types::{EntityFact, EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");
        let engine = StorageEngine::open(&path).unwrap();

        // Create initial profile
        let entity_id = EntityId::new();
        let mut profile = EntityProfile::new(
            entity_id.clone(),
            "Alice".to_string(),
            "person".to_string(),
        );
        profile.add_fact(EntityFact::new(
            "occupation",
            "engineer",
            0.9,
            MemoryId::new(),
        ));
        engine.store_entity_profile(&profile).unwrap();

        // Update profile with new fact
        profile.add_fact(EntityFact::new(
            "skill",
            "Rust",
            0.85,
            MemoryId::new(),
        ));
        engine.store_entity_profile(&profile).unwrap();

        // Retrieve and verify update
        let retrieved = engine.get_entity_profile("Alice").unwrap().unwrap();
        assert_eq!(retrieved.facts.len(), 2);
        assert!(retrieved.facts.contains_key("occupation"));
        assert!(retrieved.facts.contains_key("skill"));
    }

    #[test]
    fn test_entity_profile_persistence() {
        use crate::types::{EntityFact, EntityId, EntityProfile};

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mfdb");

        let entity_id = EntityId::new();

        // Create and store profile
        {
            let engine = StorageEngine::open(&path).unwrap();
            let mut profile = EntityProfile::new(
                entity_id.clone(),
                "Alice".to_string(),
                "person".to_string(),
            );
            profile.add_fact(EntityFact::new(
                "occupation",
                "engineer",
                0.9,
                MemoryId::new(),
            ));
            engine.store_entity_profile(&profile).unwrap();
        }

        // Reopen and verify
        {
            let engine = StorageEngine::open(&path).unwrap();
            let retrieved = engine.get_entity_profile("Alice").unwrap().unwrap();
            assert_eq!(retrieved.name, "Alice");
            assert_eq!(retrieved.facts.get("occupation").unwrap()[0].value, "engineer");
        }
    }
}
