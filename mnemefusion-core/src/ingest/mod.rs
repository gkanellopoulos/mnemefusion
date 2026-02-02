//! Ingestion module for entity extraction and memory preprocessing
//!
//! This module handles entity extraction from memory content and coordinates
//! atomic ingestion across all dimensions.
//!
//! # SLM Metadata Extraction
//!
//! When the `slm` feature is enabled, the ingestion pipeline can use a Small Language
//! Model to extract rich metadata at ingestion time. This enables fast, accurate
//! retrieval without query-time SLM inference.
//!
//! See [`slm_extractor`] for details on the extracted metadata types.

pub mod causal_extractor;
pub mod entity_extractor;
pub mod pipeline;
pub mod slm_extractor;
pub mod temporal_extractor;

pub use causal_extractor::{get_causal_extractor, CausalExtractor};
pub use entity_extractor::{EntityExtractor, SimpleEntityExtractor};
pub use pipeline::IngestionPipeline;
pub use temporal_extractor::{get_temporal_extractor, TemporalExpression, TemporalExtractor};

// SLM metadata types (always available for serialization/deserialization)
pub use slm_extractor::{
    CausalMetadata, CausalRelationship, ExtractedEntity, ExtractedEntityFact, SlmMetadata,
    TemporalMetadata,
};

// SLM extractor (only available with slm feature)
#[cfg(feature = "slm")]
pub use slm_extractor::SlmMetadataExtractor;
