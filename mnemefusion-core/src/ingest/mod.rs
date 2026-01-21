//! Ingestion module for entity extraction and memory preprocessing
//!
//! This module handles entity extraction from memory content and coordinates
//! atomic ingestion across all dimensions.

pub mod entity_extractor;
pub mod pipeline;

pub use entity_extractor::{EntityExtractor, SimpleEntityExtractor};
pub use pipeline::IngestionPipeline;
