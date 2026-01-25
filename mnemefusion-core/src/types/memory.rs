//! Memory types and identifiers
//!
//! This module defines the core Memory type and MemoryId identifier.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

use super::Timestamp;

/// Reserved metadata key for namespace
pub const NAMESPACE_METADATA_KEY: &str = "__mf_namespace__";

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
    pub fn new_with_timestamp(content: String, embedding: Vec<f32>, timestamp: Timestamp) -> Self {
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

    /// Set the source (provenance) for this memory
    ///
    /// The source is stored as JSON in a reserved metadata key.
    /// This allows backward compatibility with the v1 file format.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    /// use mnemefusion_core::types::source::{Source, SourceType};
    ///
    /// let mut memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// let source = Source::new(SourceType::Manual).with_id("test_123");
    /// memory.set_source(source).unwrap();
    ///
    /// let retrieved = memory.get_source().unwrap();
    /// assert!(retrieved.is_some());
    /// ```
    pub fn set_source(&mut self, source: super::Source) -> Result<()> {
        let json = source.to_json()?;
        self.metadata
            .insert(super::SOURCE_METADATA_KEY.to_string(), json);
        Ok(())
    }

    /// Get the source (provenance) for this memory, if it exists
    ///
    /// Returns None if no source was set, or an error if the source
    /// JSON is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    ///
    /// let memory = Memory::new("test content".into(), vec![0.1; 384]);
    ///
    /// let source = memory.get_source().unwrap();
    /// assert!(source.is_none());
    /// ```
    pub fn get_source(&self) -> Result<Option<super::Source>> {
        if let Some(json) = self.metadata.get(super::SOURCE_METADATA_KEY) {
            let source = super::Source::from_json(json)?;
            Ok(Some(source))
        } else {
            Ok(None)
        }
    }

    /// Remove the source from this memory
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    /// use mnemefusion_core::types::source::{Source, SourceType};
    ///
    /// let mut memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// let source = Source::new(SourceType::Manual);
    /// memory.set_source(source).unwrap();
    ///
    /// memory.clear_source();
    /// assert!(memory.get_source().unwrap().is_none());
    /// ```
    pub fn clear_source(&mut self) {
        self.metadata.remove(super::SOURCE_METADATA_KEY);
    }

    /// Set the namespace for this memory
    ///
    /// Namespaces enable multi-user and multi-context isolation.
    /// The namespace is stored as a reserved metadata key.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace string. Empty string `""` is the default namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    ///
    /// let mut memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// memory.set_namespace("user_123");
    ///
    /// assert_eq!(memory.get_namespace(), "user_123");
    /// ```
    pub fn set_namespace(&mut self, namespace: impl Into<String>) {
        let ns = namespace.into();
        if ns.is_empty() {
            self.metadata.remove(NAMESPACE_METADATA_KEY);
        } else {
            self.metadata.insert(NAMESPACE_METADATA_KEY.to_string(), ns);
        }
    }

    /// Get the namespace for this memory
    ///
    /// Returns the namespace string, or empty string `""` if no namespace is set
    /// (which represents the default namespace).
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    ///
    /// let memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// assert_eq!(memory.get_namespace(), ""); // Default namespace
    ///
    /// let mut memory2 = Memory::new("test".into(), vec![0.1; 384]);
    /// memory2.set_namespace("org_1/user_123");
    /// assert_eq!(memory2.get_namespace(), "org_1/user_123");
    /// ```
    pub fn get_namespace(&self) -> String {
        self.metadata
            .get(NAMESPACE_METADATA_KEY)
            .cloned()
            .unwrap_or_default()
    }

    /// Clear the namespace (revert to default namespace)
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::memory::Memory;
    ///
    /// let mut memory = Memory::new("test content".into(), vec![0.1; 384]);
    /// memory.set_namespace("user_123");
    /// assert_eq!(memory.get_namespace(), "user_123");
    ///
    /// memory.clear_namespace();
    /// assert_eq!(memory.get_namespace(), ""); // Back to default
    /// ```
    pub fn clear_namespace(&mut self) {
        self.metadata.remove(NAMESPACE_METADATA_KEY);
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

        let memory = Memory::new_with_metadata("test".to_string(), vec![0.1], metadata);

        assert_eq!(memory.get_metadata("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_memory_source_integration() {
        use super::super::{Source, SourceType};

        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        // Initially no source
        assert!(memory.get_source().unwrap().is_none());

        // Set a source
        let source = Source::new(SourceType::Conversation)
            .with_id("conv_123")
            .with_confidence(0.9);
        memory.set_source(source.clone()).unwrap();

        // Retrieve and verify
        let retrieved = memory.get_source().unwrap().unwrap();
        assert_eq!(retrieved.source_type, SourceType::Conversation);
        assert_eq!(retrieved.id, Some("conv_123".to_string()));
        assert_eq!(retrieved.confidence, Some(0.9));
    }

    #[test]
    fn test_memory_source_clear() {
        use super::super::{Source, SourceType};

        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);
        let source = Source::new(SourceType::Manual);

        memory.set_source(source).unwrap();
        assert!(memory.get_source().unwrap().is_some());

        memory.clear_source();
        assert!(memory.get_source().unwrap().is_none());
    }

    #[test]
    fn test_memory_source_roundtrip() {
        use super::super::{Source, SourceType};

        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        let source = Source::new(SourceType::Document)
            .with_id("doc_456")
            .with_location("page 42")
            .with_extractor("PDFExtractor");

        memory.set_source(source.clone()).unwrap();

        // Source should be stored in metadata
        assert!(memory
            .metadata
            .contains_key(super::super::SOURCE_METADATA_KEY));

        // Should be able to retrieve it
        let retrieved = memory.get_source().unwrap().unwrap();
        assert_eq!(retrieved.source_type, source.source_type);
        assert_eq!(retrieved.id, source.id);
        assert_eq!(retrieved.location, source.location);
        assert_eq!(retrieved.extractor, source.extractor);
    }

    #[test]
    fn test_memory_namespace_default() {
        let memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        // Default namespace is empty string
        assert_eq!(memory.get_namespace(), "");
    }

    #[test]
    fn test_memory_namespace_set_and_get() {
        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        // Set namespace
        memory.set_namespace("user_123");
        assert_eq!(memory.get_namespace(), "user_123");

        // Namespace should be in metadata
        assert!(memory.metadata.contains_key(NAMESPACE_METADATA_KEY));

        // Change namespace
        memory.set_namespace("org_1/user_456");
        assert_eq!(memory.get_namespace(), "org_1/user_456");
    }

    #[test]
    fn test_memory_namespace_clear() {
        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        // Set namespace
        memory.set_namespace("user_123");
        assert_eq!(memory.get_namespace(), "user_123");

        // Clear namespace
        memory.clear_namespace();
        assert_eq!(memory.get_namespace(), "");

        // Metadata key should be removed
        assert!(!memory.metadata.contains_key(NAMESPACE_METADATA_KEY));
    }

    #[test]
    fn test_memory_namespace_empty_string() {
        let mut memory = Memory::new("test content".to_string(), vec![0.1; 384]);

        // Setting empty string should be same as clearing
        memory.set_namespace("user_123");
        memory.set_namespace("");

        assert_eq!(memory.get_namespace(), "");
        assert!(!memory.metadata.contains_key(NAMESPACE_METADATA_KEY));
    }
}
