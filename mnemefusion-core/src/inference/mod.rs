//! Native LLM inference for entity extraction
//!
//! This module provides Rust-native inference using llama.cpp,
//! eliminating the Python subprocess dependency.
//!
//! # Architecture
//!
//! - `InferenceEngine`: Low-level model loading and generation
//! - `JsonGrammar`: GBNF grammar for constrained JSON output
//! - `EntitySchema`: Schema definitions for extraction output
//!
//! # Example
//!
//! ```rust,ignore
//! use mnemefusion_core::inference::{InferenceEngine, JsonGrammar};
//!
//! let engine = InferenceEngine::load("model.gguf", -1)?;
//! let grammar = JsonGrammar::entity_extraction();
//! let json = engine.generate_with_grammar("Extract entities from: ...", &grammar)?;
//! ```

#[cfg(feature = "entity-extraction")]
mod engine;
#[cfg(feature = "entity-extraction")]
mod grammar;

#[cfg(feature = "entity-extraction")]
pub use engine::InferenceEngine;
#[cfg(feature = "entity-extraction")]
pub use grammar::JsonGrammar;
