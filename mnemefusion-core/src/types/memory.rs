//! Memory types and identifiers
//!
//! This module defines the core Memory type and MemoryId identifier.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

use super::Timestamp;

/// Unique identifier for a memory
///
/// MemoryId is a UUID-based identifier that can be converted to/from u64
/// for use with the vector index (usearch requires u64 keys).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(Uuid);

impl MemoryId {
    /// Create a new random MemoryId
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Convert MemoryId to u64 for vector index
    ///
    /// Uses the first 8 bytes of the UUID. This is sufficient for uniqueness
    /// in practice for up to millions of memories.
    pub fn to_u64(&self) -> u64 {
        let bytes = self.0.as_bytes();
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    /// Create MemoryId from u64 (for vector index results)
    ///
    /// This is a partial reconstruction - we store the full UUID in the database
    /// and use this only for index lookups.
    pub fn from_u64(val: u64) -> Self {
        let bytes = val.to_le_bytes();
        let mut uuid_bytes = [0u8; 16];
        uuid_bytes[0..8].copy_from_slice(&bytes);
        // Note: This creates a partial UUID. In practice, we'll look up the full
        // memory record from storage using this as a key.
        Self(Uuid::from_bytes(uuid_bytes))
    }

    /// Convert MemoryId to bytes for storage
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Create MemoryId from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 16 {
            return Err(Error::InvalidMemoryId(format!(
                "Expected 16 bytes, got {}",
                bytes.len()
            )));
        }
        let mut array = [0u8; 16];
        array.copy_from_slice(bytes);
        Ok(Self(Uuid::from_bytes(array)))
    }

    /// Parse MemoryId from string representation
    pub fn parse(s: &str) -> Result<Self> {
        Uuid::parse_str(s)
            .map(Self)
            .map_err(|e| Error::InvalidMemoryId(e.to_string()))
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    /// Create MemoryId from u128 (for testing)
    #[cfg(test)]
    pub fn from_u128(val: u128) -> Self {
        Self(Uuid::from_u128(val))
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A memory record containing content, embedding, and metadata
#[derive(Debug, Clone)]
pub struct Memory {
    /// Unique identifier
    pub id: MemoryId,

    /// Text content of the memory
    pub content: String,

    /// Vector embedding (dimension must match config)
    pub embedding: Vec<f32>,

    /// When the memory was created
    pub created_at: Timestamp,

    /// Arbitrary key-value metadata
    pub metadata: HashMap<String, String>,
}

impl Memory {
    /// Create a new memory with generated ID and current timestamp
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        Self {
            id: MemoryId::new(),
            content,
            embedding,
            created_at: Timestamp::now(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new memory with custom timestamp
    pub fn new_with_timestamp(
        content: String,
        embedding: Vec<f32>,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            id: MemoryId::new(),
            content,
            embedding,
            created_at: timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Create a new memory with metadata
    pub fn new_with_metadata(
        content: String,
        embedding: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id: MemoryId::new(),
            content,
            embedding,
            created_at: Timestamp::now(),
            metadata,
        }
    }

    /// Add or update a metadata field
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get a metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Validate embedding dimension
    pub fn validate_dimension(&self, expected: usize) -> Result<()> {
        let got = self.embedding.len();
        if got != expected {
            Err(Error::InvalidEmbeddingDimension { expected, got })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_id_new() {
        let id1 = MemoryId::new();
        let id2 = MemoryId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_memory_id_to_u64() {
        let id = MemoryId::new();
        let val = id.to_u64();
        assert!(val > 0);

        // Test conversion is deterministic
        assert_eq!(id.to_u64(), id.to_u64());
    }

    #[test]
    fn test_memory_id_bytes() {
        let id = MemoryId::new();
        let bytes = id.as_bytes();
        assert_eq!(bytes.len(), 16);

        let restored = MemoryId::from_bytes(bytes).unwrap();
        assert_eq!(id, restored);
    }

    #[test]
    fn test_memory_id_parse() {
        let id = MemoryId::new();
        let s = id.to_string();
        let parsed = MemoryId::parse(&s).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_memory_id_invalid_bytes() {
        let result = MemoryId::from_bytes(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_new() {
        let content = "test content".to_string();
        let embedding = vec![0.1, 0.2, 0.3];
        let memory = Memory::new(content.clone(), embedding.clone());

        assert_eq!(memory.content, content);
        assert_eq!(memory.embedding, embedding);
        assert!(memory.metadata.is_empty());
    }

    #[test]
    fn test_memory_metadata() {
        let mut memory = Memory::new("test".to_string(), vec![0.1]);
        memory.set_metadata("key1".to_string(), "value1".to_string());
        memory.set_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(memory.get_metadata("key1"), Some(&"value1".to_string()));
        assert_eq!(memory.get_metadata("key2"), Some(&"value2".to_string()));
        assert_eq!(memory.get_metadata("key3"), None);
    }

    #[test]
    fn test_memory_validate_dimension() {
        let memory = Memory::new("test".to_string(), vec![0.1, 0.2, 0.3]);

        assert!(memory.validate_dimension(3).is_ok());
        assert!(memory.validate_dimension(4).is_err());
        assert!(memory.validate_dimension(2).is_err());
    }

    #[test]
    fn test_memory_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let memory = Memory::new_with_metadata(
            "test".to_string(),
            vec![0.1],
            metadata,
        );

        assert_eq!(memory.get_metadata("source"), Some(&"test".to_string()));
    }
}
