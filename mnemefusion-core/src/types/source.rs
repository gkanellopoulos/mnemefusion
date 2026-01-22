//! Source tracking (provenance) for memories
//!
//! This module provides structured metadata for tracking where memories came from.
//! Sources are stored as JSON in the memory's metadata HashMap using a reserved key.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reserved metadata key for source information
pub const SOURCE_METADATA_KEY: &str = "__mf_source__";

/// Type of source for a memory
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    /// From a conversation or chat
    Conversation,
    /// From a document (file, PDF, etc.)
    Document,
    /// From a URL
    Url,
    /// Manually added by user
    Manual,
    /// Inferred or derived from other memories
    Inference,
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceType::Conversation => write!(f, "conversation"),
            SourceType::Document => write!(f, "document"),
            SourceType::Url => write!(f, "url"),
            SourceType::Manual => write!(f, "manual"),
            SourceType::Inference => write!(f, "inference"),
        }
    }
}

impl std::str::FromStr for SourceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "conversation" => Ok(SourceType::Conversation),
            "document" => Ok(SourceType::Document),
            "url" => Ok(SourceType::Url),
            "manual" => Ok(SourceType::Manual),
            "inference" => Ok(SourceType::Inference),
            _ => Err(format!("Invalid source type: {}", s)),
        }
    }
}

/// Structured provenance metadata for a memory
///
/// Sources track where a memory came from, providing context and traceability.
/// This is essential for explainability, debugging, and data lineage tracking.
///
/// # Examples
///
/// ```
/// use mnemefusion_core::types::source::{Source, SourceType};
///
/// // Create a conversation source
/// let source = Source::new(SourceType::Conversation)
///     .with_id("conv_123")
///     .with_confidence(0.95);
///
/// assert_eq!(source.source_type, SourceType::Conversation);
/// assert_eq!(source.id, Some("conv_123".to_string()));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Type of source
    #[serde(rename = "type")]
    pub source_type: SourceType,

    /// Optional unique identifier for the source
    /// e.g., conversation_id, document_hash, URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Optional location within the source
    /// e.g., page number, line number, timestamp in conversation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,

    /// Optional timestamp from the original source
    /// ISO 8601 format string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,

    /// Optional original text snippet (for context)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_text: Option<String>,

    /// Confidence score for inferred sources (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,

    /// Name of the extractor/system that created this source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extractor: Option<String>,

    /// Additional source-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl Source {
    /// Create a new source with required type
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::source::{Source, SourceType};
    ///
    /// let source = Source::new(SourceType::Document);
    /// assert_eq!(source.source_type, SourceType::Document);
    /// ```
    pub fn new(source_type: SourceType) -> Self {
        Self {
            source_type,
            id: None,
            location: None,
            timestamp: None,
            original_text: None,
            confidence: None,
            extractor: None,
            metadata: None,
        }
    }

    /// Builder pattern: set ID
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::source::{Source, SourceType};
    ///
    /// let source = Source::new(SourceType::Url)
    ///     .with_id("https://example.com/page");
    /// assert_eq!(source.id, Some("https://example.com/page".to_string()));
    /// ```
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Builder pattern: set location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Builder pattern: set timestamp
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }

    /// Builder pattern: set original text
    pub fn with_original_text(mut self, text: impl Into<String>) -> Self {
        self.original_text = Some(text.into());
        self
    }

    /// Builder pattern: set confidence (clamped to 0.0-1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use mnemefusion_core::types::source::{Source, SourceType};
    ///
    /// let source = Source::new(SourceType::Inference)
    ///     .with_confidence(0.95);
    /// assert_eq!(source.confidence, Some(0.95));
    ///
    /// // Confidence is clamped
    /// let source2 = Source::new(SourceType::Inference)
    ///     .with_confidence(1.5);
    /// assert_eq!(source2.confidence, Some(1.0));
    /// ```
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Builder pattern: set extractor
    pub fn with_extractor(mut self, extractor: impl Into<String>) -> Self {
        self.extractor = Some(extractor.into());
        self
    }

    /// Builder pattern: set metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Serialize to JSON string for metadata storage
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    pub fn to_json(&self) -> Result<String, crate::Error> {
        serde_json::to_string(self)
            .map_err(|e| crate::Error::InvalidSource(format!("Serialization failed: {}", e)))
    }

    /// Deserialize from JSON string
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails
    pub fn from_json(json: &str) -> Result<Self, crate::Error> {
        serde_json::from_str(json)
            .map_err(|e| crate::Error::InvalidSource(format!("Deserialization failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_type_display() {
        assert_eq!(SourceType::Conversation.to_string(), "conversation");
        assert_eq!(SourceType::Document.to_string(), "document");
        assert_eq!(SourceType::Url.to_string(), "url");
        assert_eq!(SourceType::Manual.to_string(), "manual");
        assert_eq!(SourceType::Inference.to_string(), "inference");
    }

    #[test]
    fn test_source_type_from_str() {
        use std::str::FromStr;

        assert_eq!(SourceType::from_str("conversation").unwrap(), SourceType::Conversation);
        assert_eq!(SourceType::from_str("DOCUMENT").unwrap(), SourceType::Document);
        assert_eq!(SourceType::from_str("Url").unwrap(), SourceType::Url);
        assert!(SourceType::from_str("invalid").is_err());
    }

    #[test]
    fn test_source_creation() {
        let source = Source::new(SourceType::Conversation)
            .with_id("conv_123")
            .with_confidence(0.95);

        assert_eq!(source.source_type, SourceType::Conversation);
        assert_eq!(source.id, Some("conv_123".to_string()));
        assert_eq!(source.confidence, Some(0.95));
        assert!(source.location.is_none());
    }

    #[test]
    fn test_source_confidence_clamping() {
        let source1 = Source::new(SourceType::Inference).with_confidence(1.5);
        assert_eq!(source1.confidence, Some(1.0));

        let source2 = Source::new(SourceType::Inference).with_confidence(-0.5);
        assert_eq!(source2.confidence, Some(0.0));

        let source3 = Source::new(SourceType::Inference).with_confidence(0.75);
        assert_eq!(source3.confidence, Some(0.75));
    }

    #[test]
    fn test_source_serialization() {
        let source = Source::new(SourceType::Document)
            .with_id("doc_456")
            .with_location("page 42")
            .with_original_text("Original text snippet")
            .with_extractor("PDFExtractor v1.0");

        let json = source.to_json().unwrap();
        let restored = Source::from_json(&json).unwrap();

        assert_eq!(source.source_type, restored.source_type);
        assert_eq!(source.id, restored.id);
        assert_eq!(source.location, restored.location);
        assert_eq!(source.original_text, restored.original_text);
        assert_eq!(source.extractor, restored.extractor);
    }

    #[test]
    fn test_source_serialization_minimal() {
        let source = Source::new(SourceType::Manual);
        let json = source.to_json().unwrap();
        let restored = Source::from_json(&json).unwrap();

        assert_eq!(source.source_type, restored.source_type);
        assert!(restored.id.is_none());
        assert!(restored.location.is_none());
    }

    #[test]
    fn test_source_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), "Alice".to_string());
        meta.insert("department".to_string(), "Engineering".to_string());

        let source = Source::new(SourceType::Document).with_metadata(meta.clone());

        assert_eq!(source.metadata, Some(meta));
    }

    #[test]
    fn test_source_full_example() {
        let mut custom_meta = HashMap::new();
        custom_meta.insert("app".to_string(), "ChatBot v2.0".to_string());

        let source = Source::new(SourceType::Conversation)
            .with_id("conv_20260122_001")
            .with_location("message #42")
            .with_timestamp("2026-01-22T10:30:00Z")
            .with_original_text("User asked about source tracking")
            .with_confidence(1.0)
            .with_extractor("MemoryExtractor v1.0")
            .with_metadata(custom_meta);

        // Test serialization round-trip
        let json = source.to_json().unwrap();
        let restored = Source::from_json(&json).unwrap();

        assert_eq!(restored.source_type, SourceType::Conversation);
        assert_eq!(restored.id, Some("conv_20260122_001".to_string()));
        assert_eq!(restored.location, Some("message #42".to_string()));
        assert_eq!(restored.confidence, Some(1.0));
    }
}
