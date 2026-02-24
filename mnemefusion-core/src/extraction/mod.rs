//! Entity extraction using Qwen 3.5 SLM
//!
//! Provides SQLite-style API for entity profiling:
//! - Self-contained (model bundled or auto-downloaded)
//! - No external services required
//! - Guaranteed JSON schema compliance via grammar-constrained decoding
//!
//! # Example
//!
//! ```rust,ignore
//! use mnemefusion_core::extraction::{LlmEntityExtractor, ModelTier};
//!
//! let mut extractor = LlmEntityExtractor::load(ModelTier::Balanced)?;
//! let result = extractor.extract("Caroline is researching adoption agencies")?;
//!
//! for fact in result.entity_facts {
//!     println!("{}: {} = {}", fact.entity, fact.fact_type, fact.value);
//! }
//! ```

#[cfg(feature = "entity-extraction")]
mod extractor;
#[cfg(feature = "entity-extraction")]
mod output;
#[cfg(feature = "entity-extraction")]
mod prompt;

#[cfg(feature = "entity-extraction")]
pub use extractor::{ExtractionPerspective, LlmEntityExtractor, ModelTier};
#[cfg(feature = "entity-extraction")]
pub use output::{
    ExtractedEntity, ExtractedFact, ExtractedRelationship, ExtractionResult, TypedRecord,
};
#[cfg(feature = "entity-extraction")]
pub use prompt::{apply_chat_template, ModelFamily};
