//! Ingestion module for entity extraction and memory preprocessing
//!
//! This module handles entity extraction from memory content and coordinates
//! atomic ingestion across all dimensions.

pub mod causal_extractor;
pub mod entity_extractor;
pub mod pipeline;
pub mod temporal_extractor;

pub use causal_extractor::{get_causal_extractor, CausalExtractor};
pub use entity_extractor::{EntityExtractor, SimpleEntityExtractor};
pub use pipeline::IngestionPipeline;
pub use temporal_extractor::{get_temporal_extractor, TemporalExpression, TemporalExtractor};
