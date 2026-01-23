//! Batch operation types for efficient bulk operations
//!
//! This module provides types for batch adding and deleting memories,
//! enabling 10x+ performance improvements for bulk operations.

use super::{Memory, MemoryId, Source, Timestamp};
use std::collections::HashMap;

/// Input for a single memory in a batch operation
///
/// This is a convenience type for batch operations that allows
/// constructing memories with all fields specified upfront.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::types::batch::MemoryInput;
/// use mnemefusion_core::types::{Source, SourceType};
///
/// let input = MemoryInput {
///     content: "Meeting notes".to_string(),
///     embedding: vec![0.1; 384],
///     metadata: None,
///     timestamp: None,
///     source: Some(Source::new(SourceType::Conversation).with_id("conv_123")),
///     namespace: Some("user_123".to_string()),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MemoryInput {
    /// Text content of the memory
    pub content: String,

    /// Vector embedding (must match configured dimension)
    pub embedding: Vec<f32>,

    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,

    /// Optional custom timestamp (defaults to current time if None)
    pub timestamp: Option<Timestamp>,

    /// Optional source/provenance tracking
    pub source: Option<Source>,

    /// Optional namespace (defaults to empty string "" for default namespace)
    pub namespace: Option<String>,
}

impl MemoryInput {
    /// Create a new MemoryInput with required fields
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::batch::MemoryInput;
    ///
    /// let input = MemoryInput::new("test content".to_string(), vec![0.1; 384]);
    /// ```
    pub fn new(content: String, embedding: Vec<f32>) -> Self {
        Self {
            content,
            embedding,
            metadata: None,
            timestamp: None,
            source: None,
            namespace: None,
        }
    }

    /// Builder pattern: set metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Builder pattern: set timestamp
    pub fn with_timestamp(mut self, timestamp: Timestamp) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Builder pattern: set source
    pub fn with_source(mut self, source: Source) -> Self {
        self.source = Some(source);
        self
    }

    /// Builder pattern: set namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Convert MemoryInput to Memory with generated ID
    ///
    /// This is used internally by the batch ingestion pipeline.
    pub(crate) fn to_memory(&self) -> Memory {
        let mut memory = if let Some(ts) = self.timestamp {
            Memory::new_with_timestamp(self.content.clone(), self.embedding.clone(), ts)
        } else {
            Memory::new(self.content.clone(), self.embedding.clone())
        };

        // Add metadata if present
        if let Some(metadata) = &self.metadata {
            for (key, value) in metadata {
                memory.set_metadata(key.clone(), value.clone());
            }
        }

        // Add source if present
        if let Some(source) = &self.source {
            // Ignore error - if source serialization fails, just skip it
            let _ = memory.set_source(source.clone());
        }

        // Set namespace if present
        if let Some(namespace) = &self.namespace {
            memory.set_namespace(namespace);
        }

        memory
    }
}

/// Result of a batch add operation
///
/// Provides detailed information about what was created, including
/// any errors that occurred during processing.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::types::batch::BatchResult;
///
/// // After a batch add operation
/// // let result = engine.add_batch(inputs).unwrap();
/// // println!("Created {} memories", result.created_count);
/// // println!("Encountered {} errors", result.errors.len());
/// ```
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// IDs of successfully created memories
    pub ids: Vec<MemoryId>,

    /// Number of memories created
    pub created_count: usize,

    /// Number of duplicates detected (if deduplication enabled)
    pub duplicate_count: usize,

    /// Errors encountered during batch processing
    pub errors: Vec<BatchError>,
}

impl BatchResult {
    /// Create a new BatchResult
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            created_count: 0,
            duplicate_count: 0,
            errors: Vec::new(),
        }
    }

    /// Check if the batch operation was completely successful
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if there were any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

impl Default for BatchResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Error information for a single memory in a batch operation
#[derive(Debug, Clone)]
pub struct BatchError {
    /// Index of the memory in the input batch
    pub index: usize,

    /// Error message
    pub message: String,

    /// Optional memory ID if the error occurred after ID generation
    pub memory_id: Option<MemoryId>,
}

impl BatchError {
    /// Create a new BatchError
    pub fn new(index: usize, message: String) -> Self {
        Self {
            index,
            message,
            memory_id: None,
        }
    }

    /// Create a BatchError with a memory ID
    pub fn with_id(index: usize, message: String, memory_id: MemoryId) -> Self {
        Self {
            index,
            message,
            memory_id: Some(memory_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SourceType;

    #[test]
    fn test_memory_input_new() {
        let input = MemoryInput::new("test content".to_string(), vec![0.1; 384]);

        assert_eq!(input.content, "test content");
        assert_eq!(input.embedding.len(), 384);
        assert!(input.metadata.is_none());
        assert!(input.timestamp.is_none());
        assert!(input.source.is_none());
        assert!(input.namespace.is_none());
    }

    #[test]
    fn test_memory_input_builder() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let source = Source::new(SourceType::Manual);
        let timestamp = Timestamp::now();

        let input = MemoryInput::new("test".to_string(), vec![0.1; 384])
            .with_metadata(metadata.clone())
            .with_timestamp(timestamp)
            .with_source(source);

        assert_eq!(input.metadata, Some(metadata));
        assert_eq!(input.timestamp, Some(timestamp));
        assert!(input.source.is_some());
    }

    #[test]
    fn test_memory_input_to_memory() {
        let input = MemoryInput::new("test content".to_string(), vec![0.1; 384]);
        let memory = input.to_memory();

        assert_eq!(memory.content, "test content");
        assert_eq!(memory.embedding.len(), 384);
        assert!(memory.metadata.is_empty() || memory.metadata.len() == 0);
    }

    #[test]
    fn test_memory_input_to_memory_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("project".to_string(), "alpha".to_string());

        let input = MemoryInput::new("test".to_string(), vec![0.1; 384])
            .with_metadata(metadata.clone());

        let memory = input.to_memory();
        assert_eq!(memory.get_metadata("project"), Some(&"alpha".to_string()));
    }

    #[test]
    fn test_memory_input_to_memory_with_source() {
        let source = Source::new(SourceType::Conversation).with_id("conv_123");

        let input = MemoryInput::new("test".to_string(), vec![0.1; 384]).with_source(source);

        let memory = input.to_memory();
        let retrieved_source = memory.get_source().unwrap();
        assert!(retrieved_source.is_some());
    }

    #[test]
    fn test_batch_result_new() {
        let result = BatchResult::new();

        assert_eq!(result.ids.len(), 0);
        assert_eq!(result.created_count, 0);
        assert_eq!(result.duplicate_count, 0);
        assert_eq!(result.errors.len(), 0);
        assert!(result.is_success());
    }

    #[test]
    fn test_batch_result_with_errors() {
        let mut result = BatchResult::new();
        result.created_count = 5;
        result.errors.push(BatchError::new(3, "Test error".to_string()));

        assert_eq!(result.created_count, 5);
        assert!(!result.is_success());
        assert!(result.has_errors());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_batch_error_new() {
        let error = BatchError::new(0, "Validation failed".to_string());

        assert_eq!(error.index, 0);
        assert_eq!(error.message, "Validation failed");
        assert!(error.memory_id.is_none());
    }

    #[test]
    fn test_batch_error_with_id() {
        let id = MemoryId::new();
        let error = BatchError::with_id(1, "Storage failed".to_string(), id.clone());

        assert_eq!(error.index, 1);
        assert_eq!(error.message, "Storage failed");
        assert_eq!(error.memory_id, Some(id));
    }

    #[test]
    fn test_batch_result_default() {
        let result = BatchResult::default();
        assert!(result.is_success());
        assert_eq!(result.created_count, 0);
    }
}
